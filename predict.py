import collections
import os
import glob
from typing import Sequence

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from PIL import Image

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

FLAGS = flags.FLAGS

_LABEL_DIVISOR = 1000


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


if __name__ == "__main__":
    app.run(main)
