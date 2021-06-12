# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains loss classes used in the DeepLab model."""

import collections
from typing import Any, Dict, Text

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.model.loss import base_loss


def _create_loss_and_weight(
    loss_options: config_pb2.LossOptions.SingleLossOptions, gt_key: Text,
    pred_key: Text, weight_key: Text, **kwargs: Any) -> tf.keras.losses.Loss:
  """Creates a loss and its weight from loss options.

  Args:
    loss_options: Loss options as defined by
      config_pb2.LossOptions.SingleLossOptions or None.
    gt_key: A key to extract the ground-truth from a dictionary.
    pred_key: A key to extract the prediction from a dictionary.
    weight_key: A key to extract the per-pixel weights from a dictionary.
    **kwargs: Additional parameters to initialize the loss.

  Returns:
    A tuple of an instance of tf.keras.losses.Loss and its corresponding weight
    as an integer.

  Raises:
    ValueError: An error occurs when the loss name is not a valid loss.
  """
  if loss_options is None:
    return None, 0
  if loss_options.name == 'softmax_cross_entropy':
    return base_loss.TopKCrossEntropyLoss(
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent,
        **kwargs), loss_options.weight
  elif loss_options.name == 'l1':
    return base_loss.TopKGeneralLoss(
        base_loss.mean_absolute_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight
  elif loss_options.name == 'mse':
    return base_loss.TopKGeneralLoss(
        base_loss.mean_squared_error,
        gt_key,
        pred_key,
        weight_key,
        top_k_percent_pixels=loss_options.top_k_percent), loss_options.weight

  raise ValueError('Loss %s is not a valid loss.' % loss_options.name)


class DeepLabFamilyLoss(tf.keras.layers.Layer):
  """This class contains code to build and call losses for DeepLabFamilyLoss."""

  def __init__(
      self,
      loss_options: config_pb2.LossOptions,
      num_classes: int,
      ignore_label: int = 255):
    """Initializes the losses for Panoptic-DeepLab.

    Args:
      loss_options: Loss options as defined by config_pb2.LossOptions.
      num_classes: An integer specifying the number of classes in the dataset.
      ignore_label: An optional integer specifying the ignore label or 'None'
        (default: 255).
    """
    super(DeepLabFamilyLoss, self).__init__(name='DeepLabFamilyLoss')

    self._loss_func_and_weight_dict = collections.OrderedDict()
    self._extra_loss_names = [common.TOTAL_LOSS]

    if loss_options.HasField(common.SEMANTIC_LOSS):
      self._loss_func_and_weight_dict[
          common.SEMANTIC_LOSS] = _create_loss_and_weight(
              loss_options.semantic_loss,
              common.GT_SEMANTIC_KEY,
              common.PRED_SEMANTIC_LOGITS_KEY,
              common.SEMANTIC_LOSS_WEIGHT_KEY,
              num_classes=num_classes,
              ignore_label=ignore_label)

    if loss_options.HasField(common.CENTER_LOSS):
      self._loss_func_and_weight_dict[
          common.CENTER_LOSS] = _create_loss_and_weight(
              loss_options.center_loss, common.GT_INSTANCE_CENTER_KEY,
              common.PRED_CENTER_HEATMAP_KEY, common.CENTER_LOSS_WEIGHT_KEY)

    if loss_options.HasField(common.REGRESSION_LOSS):
      self._loss_func_and_weight_dict[
          common.REGRESSION_LOSS] = _create_loss_and_weight(
              loss_options.regression_loss, common.GT_INSTANCE_REGRESSION_KEY,
              common.PRED_OFFSET_MAP_KEY, common.REGRESSION_LOSS_WEIGHT_KEY)

    # Currently, only used for Motion-DeepLab.
    if loss_options.HasField(common.MOTION_LOSS):
      self._loss_func_and_weight_dict[
          common.MOTION_LOSS] = _create_loss_and_weight(
              loss_options.motion_loss, common.GT_FRAME_OFFSET_KEY,
              common.PRED_FRAME_OFFSET_MAP_KEY,
              common.FRAME_REGRESSION_LOSS_WEIGHT_KEY)

  def get_loss_names(self):
    # Keep track of all the keys that will be returned in self.call().
    loss_names = list(self._loss_func_and_weight_dict.keys())
    return loss_names + self._extra_loss_names

  def call(self, y_true: Dict[Text, tf.Tensor],
           y_pred: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the loss computations given ground-truth and predictions.

    The loss is computed for each sample separately. Currently, smoothed
    ground-truth labels are not supported.

    Args:
      y_true: A dictionary of tf.Tensor containing all ground-truth data to
        compute the loss. Depending on the configuration, the dict has to
        contain common.GT_SEMANTIC_KEY, and optionally
        common.GT_INSTANCE_CENTER_KEY, common.GT_INSTANCE_REGRESSION_KEY, and
        common.GT_FRAME_OFFSET_KEY.
      y_pred: A dicitionary of tf.Tensor containing all predictions to compute
        the loss. Depending on the configuration, the dict has to contain
        common.PRED_SEMANTIC_LOGITS_KEY, and optionally
        common.PRED_CENTER_HEATMAP_KEY, common.PRED_OFFSET_MAP_KEY, and
        common.PRED_FRAME_OFFSET_MAP_KEY.

    Returns:
      The loss as a dict of tf.Tensor, optionally containing the following:
      - common.SEMANTIC_LOSS: [batch].
      - common.CENTER_LOSS: [batch].
      - common.REGRESSION_LOSS: [batch].
      - common.MOTION_LOSS: [batch], the frame offset regression loss.

    Raises:
      AssertionError: If the keys of the resulting_dict do not match
        self.get_loss_names()
    """
    resulting_dict = collections.OrderedDict()
    for loss_name, func_and_weight in self._loss_func_and_weight_dict.items():
      loss_func, loss_weight = func_and_weight
      loss_value = loss_func(y_true, y_pred)
      resulting_dict[loss_name] = loss_value * loss_weight

    # Also include the total loss in the resulting_dict.
    total_loss = tf.math.accumulate_n(list(resulting_dict.values()))
    resulting_dict[common.TOTAL_LOSS] = total_loss

    if sorted(resulting_dict.keys()) != sorted(self.get_loss_names()):
      raise AssertionError(
          'The keys of the resulting_dict should match self.get_loss_names()')
    return resulting_dict
