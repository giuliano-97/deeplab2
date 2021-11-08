import argparse
import logging
import math
import re
import shutil
import tarfile

from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from deeplab2.data import data_utils

logging.basicConfig(level=logging.INFO)


_PANOPTIC_LABEL_FORMAT = "raw"
_TF_RECORD_PATTERN = "%s-%05d-of-%05d.tfrecord"
_IMAGES_DIR_NAME = "color"
_PANOPTIC_MAPS_DIR_NAME = "panoptic"
_SCANS_SEARCH_PATTERN = "scene*"
_NUM_FRAMES_SEARCH_PATTERN = r"numColorFrames\s*=\s*(?P<num_frames>\w*)"


def _load_image(image_file_path: Path):
    with tf.io.gfile.GFile(str(image_file_path), "rb") as f:
        image_data = f.read()
    return image_data


def _load_panoptic_map(panoptic_map_path: Path) -> Optional[str]:
    """Decodes the panoptic map from encoded image file.

    Args:
      panoptic_map_path: Path to the panoptic map image file.

    Returns:
      Panoptic map as an encoded int32 numpy array bytes or None if not existing.
    """
    with tf.io.gfile.GFile(str(panoptic_map_path), "rb") as f:
        panoptic_map = np.array(Image.open(f)).astype(np.int32)
    return panoptic_map.tobytes()


def _extract_tar_archive(tar_archive_path: Path):
    tar_archive = tarfile.open(str(tar_archive_path), "r:gz")
    extract_dir_path = tar_archive_path.parent
    tar_archive.extractall(
        path=str(extract_dir_path),
    )


def _load_scan_ids_from_file(scan_ids_file_path: Path) -> List[str]:
    with scan_ids_file_path.open("r") as f:
        return f.readlines()


def _find_scans(scans_root_dir: Path) -> List[str]:
    scan_dirs = scans_root_dir.glob(_SCANS_SEARCH_PATTERN)
    return [scan_dir.name for scan_dir in scan_dirs if scan_dir.is_dir()]


def _get_image_info_from_path(image_path: Path) -> Tuple[str, str]:
    """Gets image info including sequence id and image id.

    Image path is in the format of '.../scan_id/color/image_id.png',
    where `scan_id` refers to the id of the video sequence, and `image_id` is
    the id of the image in the video sequence.

    Args:
      image_path: Absolute path of the image.

    Returns:
      sequence_id, and image_id as strings.
    """
    scan_id = image_path.parents[2].name
    image_id = image_path.stem
    return scan_id, image_id


def _remove_dirs(dir_paths: List[Path]):
    for dir_path in dir_paths:
        shutil.rmtree(dir_path)


def _compute_total_number_of_frames(
    scans_root_dir_path: Path,
    scan_ids: int,
) -> int:
    cnt = 0
    for scan_id in scan_ids:
        scan_info_file_path = scans_root_dir_path / scan_id / f"{scan_id}.txt"
        if not tf.io.gfile.exists(str(scan_info_file_path)):
            logging.error(f"Scan info file missing in {scan_id}!")
            continue
        with scan_info_file_path.open("r") as f:
            matches = re.findall(_NUM_FRAMES_SEARCH_PATTERN, f.read(), re.MULTILINE)
            cnt += int(matches[0])
    return cnt


def _get_color_and_panoptic_per_shard(
    scans_root_dir_path: Path,
    scan_ids: Optional[List[str]],
    num_shards: int,
    remove_files: bool = False,
):
    if scan_ids is None:
        scan_ids = _find_scans(scans_root_dir_path)

    num_frames = _compute_total_number_of_frames(scans_root_dir_path, scan_ids)

    num_examples_per_shard = math.ceil(math.ceil(num_frames / num_shards))

    color_and_panoptic_per_shard = []
    dirs_to_remove = []
    for i, scan_id in enumerate(scan_ids):
        scan_dir_path = scans_root_dir_path / scan_id
        if not tf.io.gfile.exists(str(scan_dir_path)):
            logging.warning(f"Scan dir {str(scan_dir_path)} does not exist!")
            continue

        remove_color_files = False or remove_files
        remove_panoptic_files = False or remove_files
        images_dir_path = scan_dir_path / _IMAGES_DIR_NAME
        if not tf.io.gfile.isdir(str(images_dir_path)):
            images_archive_path = scan_dir_path / f"{_IMAGES_DIR_NAME}.tar.gz"
            if not tf.io.gfile.exists(str(images_archive_path)):
                logging.warning(f"Scan {scan_id} color images not found. Skipped.")
                continue
            _extract_tar_archive(images_archive_path)
            remove_color_files = True

        panoptic_maps_dir_path = scan_dir_path / _PANOPTIC_MAPS_DIR_NAME
        if not tf.io.gfile.exists(str(panoptic_maps_dir_path)):
            panoptic_maps_archive_path = (
                scan_dir_path / f"{_PANOPTIC_MAPS_DIR_NAME}.tar.gz"
            )
            if not tf.io.gfile.exists(str(panoptic_maps_archive_path)):
                if remove_color_files:
                    shutil.rmtree(str(images_dir_path))
                logging.warning(f"Scan {scan_id} panoptic_maps not found. Skipped.")
                continue
            _extract_tar_archive(panoptic_maps_archive_path)
            remove_panoptic_files = True

        image_file_paths = list(images_dir_path.glob("*.jpg"))
        for j, image_file_path in enumerate(image_file_paths):
            image_file_name = image_file_path.stem
            panoptic_map_file_path = panoptic_maps_dir_path / (
                re.sub(r"0+(.+)", r"\1", image_file_name) + ".png"
            )
            if tf.io.gfile.exists(str(panoptic_map_file_path)):
                logging.warning(
                    f"Panoptic map not found for image {image_file_name} in scan {scan_id}"
                )
                continue

            color_and_panoptic_per_shard.append(
                (image_file_path, panoptic_map_file_path)
            )

            shard_data = len(
                color_and_panoptic_per_shard
            ) == num_examples_per_shard or (
                # Last image of the last scan in the list
                j == len(image_file_paths)
                and i == len(scan_ids)
            )
            if shard_data:
                yield color_and_panoptic_per_shard
                color_and_panoptic_per_shard = []
                # Remove all the directories that can be removed
                _remove_dirs(dirs_to_remove)
                dirs_to_remove = []

        if remove_color_files:
            dirs_to_remove.append(images_dir_path)

        if remove_panoptic_files:
            dirs_to_remove.append(panoptic_maps_dir_path)

    # Clean up the last dirs
    _remove_dirs(dirs_to_remove)


def _create_panoptic_tfexample(
    image_path: Path,
    panoptic_map_path: Path,
) -> tf.train.Example:
    """Creates a TF example for each image.

    Args:
      image_path: Path to the image.
      panoptic_map_path: Path to the panoptic map (as an image file).

    Returns:
      TF example proto.
    """
    image_data = _load_image(image_path)
    label_data = _load_panoptic_map(panoptic_map_path)
    image_name = image_path.name
    image_format = image_path.suffix.lstrip(".").lower()
    sequence_id, frame_id = _get_image_info_from_path(image_path)
    return data_utils.create_video_tfexample(
        image_data,
        image_format,
        image_name,
        label_format=_PANOPTIC_LABEL_FORMAT,
        sequence_id=sequence_id,
        image_id=frame_id,
        label_data=label_data,
        prev_image_data=None,
        prev_label_data=None,
    )


def _create_tf_record_dataset(
    scans_root_dir_path: Path,
    dataset_tag: str,
    output_dir_path: Path,
    num_shards: int,
    scan_ids_file_path: Optional[Path],
    remove_files: bool = False,
):
    assert tf.io.gfile.isdir(str(scans_root_dir_path))

    output_dir_path.mkdir(exist_ok=True, parents=True)

    scan_ids = None
    if scan_ids_file_path is not None:
        assert tf.io.gfile.exists(str(scan_ids_file_path))
        scan_ids = _load_scan_ids_from_file(scan_ids_file_path)

    color_and_panoptic_per_shard = _get_color_and_panoptic_per_shard(
        scans_root_dir_path=scans_root_dir_path,
        scan_ids=scan_ids,
        num_shards=num_shards,
        remove_files=remove_files,
    )

    for shard_id, example_list in enumerate(color_and_panoptic_per_shard):
        shard_filename = _TF_RECORD_PATTERN % (dataset_tag, shard_id, num_shards)
        shard_file_path = output_dir_path / shard_filename
        with tf.io.TFRecordWriter(str(shard_file_path)) as tfrecord_writer:
            for image_path, panoptic_map_path in example_list:
                example = _create_panoptic_tfexample(
                    image_path,
                    panoptic_map_path,
                )
                tfrecord_writer.write(example.SerializeToString())


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Converts scans from the ScanNetV2 dataset to TFRecord",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-sd",
        "--scans_root_dir_path",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Scans root directory.",
    )

    parser.add_argument(
        "-o",
        "--output_dir_path",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to save converted TFRecord of TensorFlow examples.",
    )

    parser.add_argument(
        "-t",
        "--dataset_tag",
        type=str,
        required=True,
        help="Dataset tag. All the shards will be named as ...",
    )

    parser.add_argument(
        "-ids",
        "--scan_ids_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to a text file with the ids of the scans to consider.",
    )

    parser.add_argument(
        "-ns",
        "--num_shards",
        type=int,
        default=1000,
        help="Number of shards.",
    )

    parser.add_argument(
        "--remove_files",
        action="store_true",
        help="Remove raw images and panoptic maps.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _create_tf_record_dataset(
        scans_root_dir_path=args.scans_root_dir_path,
        dataset_tag=args.dataset_tag,
        output_dir_path=args.output_dir_path,
        scan_ids_file_path=args.scan_ids_file_path,
        num_shards=args.num_shards,
        remove_files=args.remove_files,
    )
