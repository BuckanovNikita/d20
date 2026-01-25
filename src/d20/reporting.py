"""FiftyOne dataset export functionality."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import fiftyone as fo
from loguru import logger

from d20.config import ConversionConfig
from d20.formats import get_converter
from d20.types import DatasetSplit


def export_to_fiftyone(
    dataset_format: str,
    input_path: Path | list[Path],
    config: ConversionConfig,
    split: str | None = None,
    **read_options: Any,
) -> None:
    """Export a dataset to FiftyOne App for visual inspection.

    Args:
        dataset_format: Format of the dataset (coco, yolo, voc)
        input_path: Path to input dataset (directory, file, or list of paths)
        config: Conversion configuration
        split: Specific split to export (default: all splits)
        **read_options: Additional options for reading (format-specific)

    """
    dataset_format = dataset_format.lower()
    converter = get_converter(dataset_format)

    # Read dataset using format converter
    splits = converter.read(input_path, config, **read_options)

    logger.info("Read {} split(s) from dataset", len(splits))
    for split_data in splits:
        logger.info(
            "Split '{}' has {} images and {} annotations",
            split_data.name,
            len(split_data.images),
            len(split_data.annotations),
        )
        if len(split_data.images) == 0:
            logger.warning(
                "Split '{}' has no images. This usually means the annotation file "
                "(JSON/YAML/XML) is empty or doesn't contain image entries.",
                split_data.name,
            )

    # Filter by split if specified
    if split:
        splits = [s for s in splits if s.name == split]
        if not splits:
            logger.warning("Split '{}' not found in dataset", split)
            return

    # Export each split to FiftyOne
    created_datasets = []
    for dataset_split in splits:
        if not dataset_split.images:
            logger.warning("Split '{}' has no images, skipping", dataset_split.name)
            continue
        dataset_name = f"d20-{dataset_format}-{dataset_split.name}-{uuid4().hex}"
        logger.info("Exporting {} split '{}' to FiftyOne as '{}'", dataset_format, dataset_split.name, dataset_name)

        # Create FiftyOne dataset from splits
        fo_dataset = _create_dataset_from_splits(dataset_format, dataset_split, config, dataset_name)

        # Ensure dataset is persisted
        fo_dataset.persistent = True

        # Verify dataset exists
        if fo_dataset.name in fo.list_datasets():  # type: ignore[attr-defined]
            created_datasets.append(dataset_name)
            logger.info(
                "Dataset '{}' created in FiftyOne with {} samples. Launch FiftyOne app to view: fiftyone app",
                dataset_name,
                len(dataset_split.images),
            )
        else:
            logger.error("Failed to create dataset '{}' in FiftyOne", dataset_name)

    if created_datasets:
        logger.info("Created {} dataset(s) in FiftyOne: {}", len(created_datasets), ", ".join(created_datasets))
        logger.info("To view datasets, run: fiftyone app")
        logger.info("Or open a specific dataset: fiftyone app --dataset {}", created_datasets[0])


def _create_dataset_from_splits(
    _dataset_format: str,
    dataset_split: DatasetSplit,
    config: ConversionConfig,
    dataset_name: str,
) -> fo.Dataset:  # type:ignore[name-defined]
    """Create a FiftyOne dataset from DatasetSplit objects.

    Args:
        dataset_format: Format of the dataset
        dataset_split: Dataset split to convert
        config: Conversion configuration
        dataset_name: Name for the FiftyOne dataset

    Returns:
        FiftyOne dataset

    """
    fo_dataset = fo.Dataset(dataset_name, overwrite=True)  # type: ignore[attr-defined]
    fo_dataset.persistent = True

    # Add samples to FiftyOne dataset
    sample_count = 0
    missing_images = 0
    for image_info in dataset_split.images:
        # Check if image file exists
        if not image_info.path.exists():
            logger.warning("Image file not found: {}", image_info.path)
            missing_images += 1
            continue

        # Get annotations for this image
        image_annotations = [ann for ann in dataset_split.annotations if ann.image_id == image_info.image_id]

        # Create detection objects
        detections = []
        for ann in image_annotations:
            x, y, w, h = ann.bbox
            # Convert to normalized coordinates (FiftyOne uses [0, 1])
            x_norm = x / image_info.width
            y_norm = y / image_info.height
            w_norm = w / image_info.width
            h_norm = h / image_info.height

            # Ensure coordinates are within [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            label = (
                config.class_names[ann.category_id]
                if ann.category_id < len(config.class_names)
                else f"class_{ann.category_id}"
            )

            detections.append(
                fo.Detection(  # type: ignore[attr-defined]
                    label=label,
                    bounding_box=[x_norm, y_norm, w_norm, h_norm],
                ),
            )

        # Create sample
        sample = fo.Sample(  # type: ignore[attr-defined]
            filepath=str(image_info.path),
            detections=fo.Detections(detections=detections) if detections else None,  # type: ignore[attr-defined]
        )
        fo_dataset.add_sample(sample)
        sample_count += 1

    logger.info("Added {} samples to dataset '{}' ({} images were missing)", sample_count, dataset_name, missing_images)
    if sample_count == 0:
        logger.error("No samples were added to dataset '{}'. Check image paths.", dataset_name)
    return fo_dataset
