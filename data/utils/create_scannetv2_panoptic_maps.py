import functools
import os
import multiprocessing
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from PIL import Image

from scannetv2_label_conversion import SCANNETV2_TO_NYU40

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="scans_root_dir_path",
    default=None,
    help="Path to a directory containing scans as subdirs.",
)
flags.DEFINE_bool(
    name="is_scannet_frames_25k",
    default=False,
    help="Convert the scannetv2 ids to NYU40",
)
flags.DEFINE_bool(
    name="remove_semantic_and_instance",
    default=False,
    help="Semantic and instance labels will be removed.",
)
flags.DEFINE_bool(
    name="compress",
    default=False,
    help="Panoptic maps will be compressed into a tar.gz archive.",
)

flags.DEFINE_integer(name="jobs", default=2, help="Number of parallel jobs")

# Same as Cityscapes
_LABEL_DIVISOR = 255
_SEMANTIC_MAPS_ARCHIVE_SUFFIX = "_2d-label-filt.zip"
_INSTANCE_MAPS_ARCHIVE_SUFFIX = "_2d-instance-filt.zip"
_SEMANTIC_MAPS_DIR_NAME = "label-filt"
_INSTANCE_MAPS_DIR_NAME = "instance-filt"
_SEMANTIC_MAPS_DIR_NAME_SCANNET_FRAMES_25K = "label"
_INSTANCE_MAPS_DIR_NAME_SCANNET_FRAMES_25K = "instance"

_PANOPTIC_MAPS_DIR_NAME = "panoptic"

_NYU40_STUFF_CLASSES = [1, 2, 22]
_SCANNET_25K_INSTANCE_DIVISOR = 1000


def _scan_has_panoptic(scan_dir_path: Path):
    panoptic_maps_dir_path = scan_dir_path / _PANOPTIC_MAPS_DIR_NAME
    if panoptic_maps_dir_path.exists() and any(panoptic_maps_dir_path.iterdir()):
        return True

    panoptic_maps_archive_path = panoptic_maps_dir_path.with_suffix(".tar.gz")
    if panoptic_maps_archive_path.exists():
        return True

    return False


def _extract_zip_archive(path_to_zip_archive: Path):
    archive = zipfile.ZipFile(str(path_to_zip_archive))
    extract_dir = path_to_zip_archive.parent
    archive.extractall(str(extract_dir))


def _convert_semantic_map_labels(
    semantic_map: np.ndarray,
    label_conversion_dict: Dict,
):
    return np.vectorize(label_conversion_dict.get)(semantic_map)


def _decode_scannet_frames_25k_instance_map(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
):
    return instance_map - _SCANNET_25K_INSTANCE_DIVISOR * semantic_map


def _normalize_instance_map(
    instance_map: np.ndarray, semantic_map: np.array, stuff_classes: List[int]
):
    normalized_instance_map = np.zeros_like(instance_map)

    for class_id in np.unique(semantic_map):
        if class_id == 0 or class_id in stuff_classes:
            continue
        # Get mask for the current class
        class_mask = semantic_map == class_id

        # Get all the unique instance ids
        instance_ids = np.unique(instance_map[class_mask]).tolist()

        # Remove 0 just in case
        try:
            instance_ids.remove(0)
        except ValueError:
            pass

        # Remap instance ids so they are 1-indexed
        new_instance_ids = list(range(1, len(instance_ids) + 1))

        # Create new instance
        for i, old_id in enumerate(instance_ids):
            instance_mask = np.logical_and(class_mask, instance_map == old_id)
            normalized_instance_map[instance_mask] = new_instance_ids[i]

    return normalized_instance_map


def _make_panoptic_from_semantic_and_instance(
    semantic_map: np.ndarray,
    instance_map: np.ndarray,
):
    panoptic_map = semantic_map * _LABEL_DIVISOR + instance_map
    return panoptic_map.astype(np.int32)


def generate_deeplab2_panoptic_map(
    semantic_map_file_path: Path,
    instance_map_file_path: Path,
    panoptic_maps_dir_path: Path,
    is_scannet_frames_25k: bool,
):
    semantic_map = np.array(Image.open(str(semantic_map_file_path)))
    instance_map = np.array(Image.open(str(instance_map_file_path)))

    # Convert semantic labels to the target labels set
    if not is_scannet_frames_25k:
        semantic_map = _convert_semantic_map_labels(
            semantic_map,
            SCANNETV2_TO_NYU40,
        )

    if is_scannet_frames_25k:
        instance_map = _decode_scannet_frames_25k_instance_map(
            instance_map, semantic_map
        )

    # Normalize the instance map so that all the instance ids are between 1 and #instances
    if not is_scannet_frames_25k:
        instance_map = _normalize_instance_map(
            instance_map,
            semantic_map,
            _NYU40_STUFF_CLASSES,
        )

    # Make panoptic map
    panoptic_map = _make_panoptic_from_semantic_and_instance(
        semantic_map,
        instance_map,
    )

    # Save panoptic map to disk
    panoptic_map_file_path = panoptic_maps_dir_path / semantic_map_file_path.name
    panoptic_map_image = Image.fromarray(panoptic_map)
    panoptic_map_image.save(str(panoptic_map_file_path))


def _create_panoptic_maps_for_scan(
    scan_dir_path: Path,
    is_scannet_frames_25k: bool,
    remove_semantic_and_instance: bool,
    compress: bool,
):
    semantic_maps_dir_name = (
        _SEMANTIC_MAPS_DIR_NAME_SCANNET_FRAMES_25K
        if is_scannet_frames_25k
        else _SEMANTIC_MAPS_DIR_NAME
    )

    instance_maps_dir_name = (
        _INSTANCE_MAPS_DIR_NAME_SCANNET_FRAMES_25K
        if is_scannet_frames_25k
        else _INSTANCE_MAPS_DIR_NAME
    )

    # Check if panoptic maps have already been created for this scans
    if _scan_has_panoptic(scan_dir_path):
        logging.warning(f"{scan_dir_path.name} already has panoptic!")
        return
    panoptic_maps_dir_path = scan_dir_path / _PANOPTIC_MAPS_DIR_NAME
    panoptic_maps_dir_path.mkdir(exist_ok=True)

    semantic_maps_dir_path = scan_dir_path / semantic_maps_dir_name
    instance_maps_dir_path = scan_dir_path / instance_maps_dir_name
    remove_semantic = False or remove_semantic_and_instance
    remove_instance = False or remove_semantic_and_instance
    if not semantic_maps_dir_path.exists():
        # If not found, try to extract the archive
        semantic_maps_archive_path = scan_dir_path / (
            scan_dir_path.stem + _SEMANTIC_MAPS_ARCHIVE_SUFFIX
        )
        if not semantic_maps_archive_path.exists():
            logging.warning(
                '"label-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            return
        _extract_zip_archive(semantic_maps_archive_path)
        remove_semantic = True
    if not instance_maps_dir_path.exists():
        instance_maps_archive_path = scan_dir_path / (
            scan_dir_path.stem + _INSTANCE_MAPS_ARCHIVE_SUFFIX
        )
        if not instance_maps_archive_path.exists():
            logging.warning(
                '"instance-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            return
        _extract_zip_archive(instance_maps_archive_path)
        remove_instance = True

    semantic_map_files = sorted(list(semantic_maps_dir_path.glob("*.png")))
    instance_map_files = sorted(list(instance_maps_dir_path.glob("*.png")))

    # Generate panoptic maps
    for semantic_map_file_path, instance_map_file_path in zip(
        semantic_map_files, instance_map_files
    ):
        generate_deeplab2_panoptic_map(
            semantic_map_file_path,
            instance_map_file_path,
            panoptic_maps_dir_path,
            is_scannet_frames_25k,
        )

    # Delete semantic and instance maps
    if remove_semantic:
        shutil.rmtree(semantic_maps_dir_path)

    if remove_instance:
        shutil.rmtree(instance_maps_dir_path)

    # Compress panoptic maps into a tar.gz archive
    if compress:
        cmd = f"cd {str(scan_dir_path.absolute())} && tar --remove-files -czf {_PANOPTIC_MAPS_DIR_NAME}.tar.gz {_PANOPTIC_MAPS_DIR_NAME}"
        os.system(cmd)


def create_scannetv2_panoptic_maps(_):
    # Validate input args
    scans_root_dir_path = Path(FLAGS.scans_root_dir_path)
    assert scans_root_dir_path.exists()
    n_jobs = FLAGS.jobs
    assert n_jobs > 0
    remove_semantic_and_instance = FLAGS.remove_semantic_and_instance
    compress = FLAGS.compress
    is_scannet_frames_25k = FLAGS.is_scannet_frames_25k

    # Get all the scan dirs
    scan_dir_paths = [
        p for p in sorted(list(scans_root_dir_path.glob("scene*"))) if p.is_dir()
    ]

    if n_jobs > 1:
        # Create panoptic maps for every directory in parallel
        job_fn = functools.partial(
            _create_panoptic_maps_for_scan,
            is_scannet_frames_25k=is_scannet_frames_25k,
            remove_semantic_and_instance=remove_semantic_and_instance,
            compress=compress,
        )
        with multiprocessing.Pool(processes=n_jobs) as p:
            p.map(job_fn, scan_dir_paths)
    else:
        for scan_dir_path in scan_dir_paths:
            _create_panoptic_maps_for_scan(
                scan_dir_path=scan_dir_path,
                is_scannet_frames_25k=is_scannet_frames_25k,
                remove_semantic_and_instance=remove_semantic_and_instance,
                compress=compress,
            )


if __name__ == "__main__":
    app.run(create_scannetv2_panoptic_maps)
