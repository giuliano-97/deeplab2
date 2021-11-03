import functools
from pathlib import Path
import numpy as np
import multiprocessing
import pandas as pd
import shutil
import zipfile

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
flags.DEFINE_bool(
    name="remove_semantic_and_instance",
    default=False,
    help="If true, semantic and instance labels will be removed.",
)

flags.DEFINE_integer(name="jobs", default=2, help="Number of parallel jobs")

# Same as Cityscapes
_LABEL_DIVISOR = 255
_SEMANTIC_MAPS_ARCHIVE_SUFFIX = "_2d-label-filt.zip"
_SEMANTIC_MAPS_DIR_NAME = "label-filt"
_INSTANCE_MAPS_ARCHIVE_SUFFIX = "_2d-instance-filt.zip"
_INSTANCE_MAPS_DIR_NAME = "instance-filt"


def extract_zip_archive(path_to_zip_archive: Path):
    archive = zipfile.ZipFile(str(path_to_zip_archive))
    extract_dir = path_to_zip_archive.parent
    archive.extractall(str(extract_dir))


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


def make_panoptic_from_semantic_and_instance(
    semantic_map: np.ndarray,
    instance_map: np.ndarray,
):
    panoptic_map = semantic_map * _LABEL_DIVISOR + instance_map
    return panoptic_map.astype(np.int32)


def generate_deeplab2_panoptic_map(
    semantic_map_file_path: Path,
    instance_map_file_path: Path,
    panoptic_maps_dir_path: Path,
    label_conversion_dict: Dict,
):
    semantic_map = np.array(Image.open(str(semantic_map_file_path)))
    instance_map = np.array(Image.open(str(instance_map_file_path)))

    # Convert semantic labels to the target labels set
    converted_semantic_map = convert_semantic_map_labels(
        semantic_map, label_conversion_dict
    )

    # Normalize the instance map so that all the instance ids are between 1 and #instances
    normalized_instanced_map = normalize_instance_map(instance_map)

    # Make panoptic map
    panoptic_map = make_panoptic_from_semantic_and_instance(
        converted_semantic_map, normalized_instanced_map
    )

    # Save panoptic map to disk
    panoptic_map_file_path = panoptic_maps_dir_path / semantic_map_file_path.name
    panoptic_map_image = Image.fromarray(panoptic_map)
    panoptic_map_image.save(str(panoptic_map_file_path))


def create_scannetv2_panoptic_maps(_):
    # Validate input args
    scans_root_dir_path = Path(FLAGS.scans_root_dir_path)
    assert scans_root_dir_path.exists()
    labels_tsv_path = Path(FLAGS.labels_tsv_path)
    assert labels_tsv_path.exists()
    n_jobs = FLAGS.jobs
    assert n_jobs > 0
    remove_semantic_and_instance = FLAGS.remove_semantic_and_instance

    # Load the labels conversion table - use the scannetv2 id as index
    label_conversion_master_table = pd.read_csv(str(labels_tsv_path), sep="\t")
    label_conversion_dict = create_label_conversion_dict(
        label_conversion_master_table, FLAGS.labels_set
    )

    # Loop over all the scans
    for scan_dir_path in scans_root_dir_path.iterdir():
        if not scan_dir_path.is_dir():
            continue

        # Check if panoptic maps have already been created for this scans
        panoptic_maps_dir_path = scan_dir_path / "panoptic"
        if panoptic_maps_dir_path.exists() and any(panoptic_maps_dir_path.iterdir()):
            continue
        panoptic_maps_dir_path.mkdir(exist_ok=True)

        semantic_maps_dir_path = scan_dir_path / _SEMANTIC_MAPS_DIR_NAME
        instance_maps_dir_path = scan_dir_path / _INSTANCE_MAPS_DIR_NAME
        remove_semantic = False or remove_semantic_and_instance
        remove_instance = False or remove_semantic_and_instance
        if not semantic_maps_dir_path.exists():
            # If not found, try to extract the archive
            semantic_maps_archive_path = scan_dir_path / (
                scan_dir_path.stem + _SEMANTIC_MAPS_ARCHIVE_SUFFIX
            )
            if not semantic_maps_archive_path.exists():
                logging.warn(
                    '"label-filt" missing in scan {}. Skipped.'.format(
                        str(scan_dir_path.name)
                    )
                )
                continue
            extract_zip_archive(semantic_maps_archive_path)
            remove_semantic = True
        if not instance_maps_dir_path.exists():
            instance_maps_archive_path = scan_dir_path / (
                scan_dir_path.stem + _INSTANCE_MAPS_ARCHIVE_SUFFIX
            )
            if not instance_maps_archive_path.exists():
                logging.warn(
                    '"instance-filt" missing in scan {}. Skipped.'.format(
                        str(scan_dir_path.name)
                    )
                )
                continue
            extract_zip_archive(instance_maps_archive_path)
            remove_instance = True

        semantic_map_files = sorted(list(semantic_maps_dir_path.glob("*.png")))
        instance_map_files = sorted(list(instance_maps_dir_path.glob("*.png")))

        # Generate panoptic maps for all files with multithreading
        job_fn = functools.partial(
            generate_deeplab2_panoptic_map,
            panoptic_maps_dir_path=panoptic_maps_dir_path,
            label_conversion_dict=label_conversion_dict,
        )

        with multiprocessing.Pool(processes=n_jobs) as p:
            p.starmap(job_fn, list(zip(semantic_map_files, instance_map_files)))

        # Delete semantic and instance maps
        if remove_semantic:
            shutil.rmtree(semantic_maps_dir_path)

        if remove_instance:
            shutil.rmtree(instance_maps_dir_path)


if __name__ == "__main__":
    app.run(create_scannetv2_panoptic_maps)
