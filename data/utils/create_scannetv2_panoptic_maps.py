from pathlib import Path
import numpy as np
import pandas as pd

from typing import Dict

from absl import app
from absl import flags
from absl import logging
from PIL import Image


FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="labels_tsv_path",
    default=None,
    help="Path to the combined-labels.tsv file of the ScanNetV2 dataset",
)
flags.DEFINE_enum(
    name="labels_set",
    default="nyu40id",
    enum_values=["nyu40id"],
    help="Name of the dataset",
)
flags.DEFINE_string(
    name="scans_root_dir_path",
    default=None,
    help="Path to a directory containing scans as subdirs.",
)

# Same as Cityscapes
_LABEL_DIVISOR = 255


def convert_semantic_map_labels(semantic_map: np.ndarray, label_conversion_dict: Dict):
    return np.vectorize(label_conversion_dict.get)(semantic_map)


def create_label_conversion_dict(label_conversion_table: pd.DataFrame, labels_set: str):
    scannetv2_label_ids = label_conversion_table["id"].tolist()
    target_label_ids = label_conversion_table[labels_set].tolist()

    # Match ids and create ids
    label_conversion_dict = dict(zip(scannetv2_label_ids, target_label_ids))

    # Add zero
    label_conversion_dict[0] = 0

    return label_conversion_dict


def normalize_instance_map(instance_map: np.ndarray):
    instance_ids = np.unique(instance_map).tolist()
    # Remove 0 if present
    try:
        instance_ids.remove(0)
    except ValueError:
        pass

    # Generate new instance ids
    new_instance_ids = [i for i in range(1, len(instance_ids) + 1)]

    # Create conversion dict
    conversion_dict = dict(zip(instance_ids, new_instance_ids))
    conversion_dict[0] = 0

    # Now convert the instance map
    return np.vectorize(conversion_dict.get)(instance_map)


def create_deeplab2_panoptic_map(semantic_map: np.ndarray, instance_map: np.ndarray):
    panoptic_map = semantic_map * _LABEL_DIVISOR + instance_map
    return panoptic_map.astype(np.int32)


def create_scannetv2_panoptic_maps(_):
    # Validate input args
    scans_root_dir_path = Path(FLAGS.scans_root_dir_path)
    assert scans_root_dir_path.exists()
    labels_tsv_path = Path(FLAGS.labels_tsv_path)
    assert labels_tsv_path.exists()

    # Load the labels conversion table - use the scannetv2 id as index
    label_conversion_master_table = pd.read_csv(str(labels_tsv_path), sep="\t")
    label_conversion_dict = create_label_conversion_dict(
        label_conversion_master_table, FLAGS.labels_set
    )

    # Loop over all the scans
    for scan_dir_path in scans_root_dir_path.iterdir():
        if not scan_dir_path.is_dir():
            continue
        semantic_maps_dir_path = scan_dir_path / "label-filt"
        instance_maps_dir_path = scan_dir_path / "instance-filt"
        if not semantic_maps_dir_path.exists():
            logging.warn(
                '"label-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            continue
        if not instance_maps_dir_path.exists():
            logging.warn(
                '"instance-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            continue

        panoptic_maps_dir_path = scan_dir_path / "panoptic"
        panoptic_maps_dir_path.mkdir(exist_ok=True)
        for semantic_map_file_path in semantic_maps_dir_path.glob("*.png"):
            instance_map_file_path = (
                instance_maps_dir_path / semantic_map_file_path.name
            )
            if not instance_map_file_path.exists():
                continue
            # Load semantic and instance map
            semantic_map = np.array(Image.open(str(semantic_map_file_path)))
            instance_map = np.array(Image.open(str(instance_map_file_path)))

            # Convert the semantic labels
            nyu40_semantic_map = convert_semantic_map_labels(
                semantic_map, label_conversion_dict
            )

            # Normalize the instance map so that all the instance ids are between 1 and #instances
            normalized_instanced_map = normalize_instance_map(instance_map)

            # Create panoptic map
            panoptic_map = create_deeplab2_panoptic_map(
                nyu40_semantic_map, normalized_instanced_map
            )

            # Save the panoptic map as png
            panoptic_map_file_path = (
                panoptic_maps_dir_path / semantic_map_file_path.name
            )
            panoptic_map_image = Image.fromarray(panoptic_map)
            panoptic_map_image.save(str(panoptic_map_file_path))


if __name__ == "__main__":
    app.run(create_scannetv2_panoptic_maps)
