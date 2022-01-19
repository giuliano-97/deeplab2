import collections
import glob
import json
import os
from typing import List, Sequence

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from PIL import Image
from scipy.stats import entropy

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags.DEFINE_string(
    "saved_model_path",
    default="",
    help="Path to the saved_model directory.",
)

flags.DEFINE_string(
    "images_dir_path",
    default="",
    help="Directory containing the target images.",
)

flags.DEFINE_string(
    "output_path",
    default="",
    help="Output directory where predictions should be saved.",
)

flags.DEFINE_boolean(
    "estimate_semantic_uncertainty",
    default=False,
    help="Whether pixel wise semantic uncertainty should be estimated from probs.",
)

flags.DEFINE_boolean(
    "estimate_instance_scores",
    default=False,
    help="Whether instance scores should be estimated from mask logits softmax.",
)

flags.DEFINE_boolean(
    "save_raw_semantic_probs",
    default=False,
    help="Save semantic probabilities to disk as .npy files.",
)

FLAGS = flags.FLAGS

_LABEL_DIVISOR = 1000
_ENTROPY_LOG_BASE = 2  # as usually done in information theory
_STUFF_CLASSES = [1, 2, 22]
_TRANSFOMER_CLASS_CONFIDENCE_THRESH = 0.7
_PIXEL_CONFIDENCE_THRESH = 0.4


def convert_prediction_to_rgb(panoptic_prediction):
    """
    Convert a panoptic prediction to RGB.
    """
    rgb = np.zeros(
        shape=(panoptic_prediction.shape[0], panoptic_prediction.shape[1], 3),
        dtype=np.uint8,
    )
    # R = semantic id
    rgb[:, :, 0] = panoptic_prediction // _LABEL_DIVISOR
    # G = instance id
    rgb[:, :, 1] = panoptic_prediction % _LABEL_DIVISOR
    # B is just zeros

    return rgb


def resize_tensor_bilinear(input_tensor, target_size):
    return tf.compat.v1.image.resize(
        input_tensor,
        target_size,
        method=tf.compat.v1.image.ResizeMethod.BILINEAR,
        align_corners=True,
        name="resize_align_corners",
    )


def estimate_semantic_uncertainty_map_from_probs(semantic_probs: np.ndarray):
    # Entropy is bounded by the cardinality of the random variable
    # i.e. in this case the number of classes
    max_entropy = np.log2(semantic_probs.shape[2])
    # Compute entropy
    semantic_probs_entropy = entropy(
        pk=semantic_probs,
        qk=None,  # No KL divergence
        base=_ENTROPY_LOG_BASE,
        axis=2,
    )
    # Normalize the entropy
    semantic_uncertainty_map = semantic_probs_entropy / max_entropy
    return semantic_uncertainty_map


def get_transformer_class_prediction(
    transformer_class_probs: tf.Tensor,
    transformer_class_confidence_threshold: float,
):
    transformer_class_pred = tf.cast(
        tf.argmax(transformer_class_probs, axis=-1), tf.float32
    )
    transformer_class_confidence = tf.reduce_max(
        transformer_class_probs, axis=-1, keepdims=False
    )
    # Filter mask IDs with class confidence less than the threshold.
    thresholded_mask = tf.cast(
        tf.greater_equal(
            transformer_class_confidence, transformer_class_confidence_threshold
        ),
        tf.float32,
    )
    transformer_class_confidence = transformer_class_confidence * thresholded_mask

    detected_mask_indices = tf.where(tf.greater(thresholded_mask, 0.5))[:, 0]
    detected_mask_class_pred = tf.gather(transformer_class_pred, detected_mask_indices)
    num_detections = tf.shape(detected_mask_indices)[0]
    return detected_mask_class_pred, detected_mask_indices, num_detections


def estimate_instance_scores(
    panoptic_pred: np.ndarray,
    pixel_space_mask_logits: tf.Tensor,
    transformer_class_logits: tf.Tensor,
):
    image_shape = panoptic_pred.shape

    # Compute class probs for each mask
    transformer_class_probs = tf.nn.softmax(transformer_class_logits, axis=-1)
    transformer_class_probs = transformer_class_probs[..., :-1]
    _, detected_mask_indices, _ = get_transformer_class_prediction(
        transformer_class_probs,
        _TRANSFOMER_CLASS_CONFIDENCE_THRESH,
    )

    # Resize to native resolution
    pixel_mask_logits = tf.compat.v1.image.resize_bilinear(
        pixel_space_mask_logits,
        (image_shape[0] + 1, image_shape[1] + 1),
        align_corners=True,
    )

    detected_pixel_mask_logits = tf.gather(
        pixel_mask_logits, detected_mask_indices, axis=-1
    )

    # Compute mask confidence for each pixel
    pixel_mask_confidence_map = tf.squeeze(
        tf.reduce_max(
            tf.nn.softmax(detected_pixel_mask_logits, axis=-1),
            axis=-1,
        )
    )

    # Drop the last row and column to match the image shape and conv to numpy
    pixel_mask_confidence_map = pixel_mask_confidence_map[:-1, :-1].numpy()

    # Loop over segments in instance prediction
    segment_ids = np.unique(panoptic_pred)
    instance_scores = []
    for segment_id in segment_ids:
        # Skip stuff segments
        segment_instance_id = segment_id % _LABEL_DIVISOR
        if segment_instance_id == 0:
            continue
        segment_class_id = segment_id // _LABEL_DIVISOR
        # Compute the score for this instance by averaging the foreground map probs
        score = np.mean(pixel_mask_confidence_map[panoptic_pred == segment_id])
        instance_scores.append(
            {
                "class_id": int(segment_class_id),
                "instance_id": int(segment_instance_id),
                "score": float(score),
            }
        )

    return instance_scores


def main(argv: Sequence[str]) -> None:

    assert tf.io.gfile.isdir(FLAGS.saved_model_path)
    assert FLAGS.images_dir_path != FLAGS.output_path
    tf.io.gfile.makedirs(FLAGS.output_path)

    if FLAGS.save_raw_semantic_probs:
        print("Saving raw semantic probs may occupy a large amount of storage.")

    # Load the model
    model = tf.saved_model.load(FLAGS.saved_model_path)

    # Collect all the images in the images directory
    image_files = glob.glob(os.path.join(FLAGS.images_dir_path, "*.jpg"))

    # Run inference over all the images found
    for image_file in image_files:
        with tf.io.gfile.GFile(image_file, "rb") as f:
            image = np.array(Image.open(f))
        prediction = model(tf.cast(image, tf.uint8))
        # Extract panoptic prediction
        panoptic_prediction = tf.squeeze(prediction["panoptic_pred"]).numpy()

        # Save the prediction to disk as two-channel png
        panoptic_prediction_file_path = os.path.join(
            FLAGS.output_path,
            os.path.splitext(os.path.basename(image_file))[0] + ".png",
        )
        panoptic_prediction_rgb = convert_prediction_to_rgb(panoptic_prediction)
        Image.fromarray(panoptic_prediction_rgb).save(panoptic_prediction_file_path)

        # Estimate the uncertainty of semantic labels
        if FLAGS.estimate_semantic_uncertainty:
            semantic_probs = tf.squeeze(prediction["semantic_probs"]).numpy()
            uncertainty_map = estimate_semantic_uncertainty_map_from_probs(
                semantic_probs
            )
            # Save the uncertainty map as tiff
            uncertainty_map_file_path = os.path.join(
                FLAGS.output_path,
                os.path.splitext(os.path.basename(image_file))[0] + "_uncertainty.tiff",
            )
            Image.fromarray(uncertainty_map).save(uncertainty_map_file_path)

        if FLAGS.save_raw_semantic_probs:
            semantic_probs = tf.squeeze(prediction["semantic_probs"]).numpy()
            semantic_probs_file_path = os.path.join(
                FLAGS.output_path,
                os.path.splitext(os.path.basename(image_file))[0]
                + "_semantic_probs.npy",
            )
            np.save(semantic_probs_file_path, semantic_probs)

        if FLAGS.estimate_instance_scores:
            instance_scores = estimate_instance_scores(
                panoptic_prediction,
                prediction["pixel_space_mask_logits"],
                prediction["transformer_class_logits"],
            )
            instance_scores_file = os.path.join(
                FLAGS.output_path,
                os.path.splitext(os.path.basename(image_file))[0]
                + "_instance_scores.json",
            )
            with open(instance_scores_file, "w") as f:
                json.dump(instance_scores, f)


if __name__ == "__main__":
    app.run(main)
