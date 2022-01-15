import collections
import glob
import logging
import os
from typing import Sequence

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from PIL import Image
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)

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
    help="Whether uncertainty should be computed from semantic probs.",
)

flags.DEFINE_boolean(
    "save_raw_semantic_probs",
    default=False,
    help="Save semantic probabilities to disk as .npy files.",
)

FLAGS = flags.FLAGS

_LABEL_DIVISOR = 1000
_ENTROPY_LOG_BASE = 2  # as usually done in information theory


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


def main(argv: Sequence[str]) -> None:

    assert tf.io.gfile.isdir(FLAGS.saved_model_path)
    assert FLAGS.images_dir_path != FLAGS.output_path
    tf.io.gfile.makedirs(FLAGS.output_path)

    if FLAGS.save_raw_semantic_probs:
        logging.warning("Saving raw semantic probs may occupy a large amount of storage.")

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
        panoptic_prediction = prediction["panoptic_pred"][0]
        panoptic_prediction_image = Image.fromarray(
            convert_prediction_to_rgb(panoptic_prediction.numpy()),
        )
        # Save the prediction to disk as two-channel png
        panoptic_prediction_file_path = os.path.join(
            FLAGS.output_path,
            os.path.splitext(os.path.basename(image_file))[0] + ".png",
        )
        panoptic_prediction_image.save(panoptic_prediction_file_path)

        # Estimate the uncertainty of semantic labels
        if FLAGS.estimate_semantic_uncertainty:
            semantic_probs = prediction["semantic_probs"][0].numpy()
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
            # Normalize the entropy and use it as proxy for uncertainty
            uncertainty = semantic_probs_entropy / max_entropy
            # Save the uncertainty map as tiff
            uncertainty_map_file_path = os.path.join(
                FLAGS.output_path,
                os.path.splitext(os.path.basename(image_file))[0] + "_uncertainty.tiff",
            )
            np.save(uncertainty_map_file_path, uncertainty)

        if FLAGS.save_raw_semantic_probs:
            semantic_probs = prediction["semantic_probs"][0]
            semantic_probs_file_path = os.path.join(
                FLAGS.output_path,
                os.path.splitext(os.path.basename(image_file))[0]
                + "_semantic_probs.npy",
            )
            np.save(semantic_probs_file_path, semantic_probs.numpy())


if __name__ == "__main__":
    app.run(main)
