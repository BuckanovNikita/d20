"""FiftyOne dataset export functionality."""

from uuid import uuid4

import fiftyone as fo
from loguru import logger

from d20.formats import get_converter
from d20.types import (
    CocoDetectedParams,
    DetectedParams,
    ExportOptions,
    Split,
    VocDetectedParams,
    YoloDetectedParams,
)


def _get_format_from_params(params: DetectedParams) -> str:
    """Get format name from DetectedParams type.

    Args:
        params: DetectedParams instance

    Returns:
        Format name (e.g., 'yolo', 'coco', 'voc')

    """
    if isinstance(params, YoloDetectedParams):
        return "yolo"
    if isinstance(params, CocoDetectedParams):
        return "coco"
    if isinstance(params, VocDetectedParams):
        return "voc"
    msg = f"Unknown DetectedParams type: {type(params)}"
    raise ValueError(msg)


def export_fiftyone(
    params: DetectedParams,
    options: ExportOptions,
) -> None:
    """Export dataset to FiftyOne App for visual inspection.

    Uses DetectedParams from autodetect() and export-specific options.
    Internally calls read() to get Dataset, then exports to FiftyOne.

    Args:
        params: DetectedParams from converter.autodetect() (contains format info)
        options: Export-specific options (split name, etc.)

    """
    # Determine format from params type
    format_name = _get_format_from_params(params)
    converter = get_converter(format_name)

    # Read dataset using DetectedParams
    dataset = converter.read(params)

    logger.info(f"Read {len(dataset.splits)} split(s) from dataset")
    for split_name, split_data in dataset.splits.items():
        logger.info(
            f"Split '{split_name}' has {len(split_data.images)} images and {len(split_data.annotations)} annotations"
        )
        if len(split_data.images) == 0:
            logger.warning(
                f"Split '{split_name}' has no images. This usually means the annotation file "
                "(JSON/YAML/XML) is empty or doesn't contain image entries."
            )

    # Filter splits if needed
    splits_to_export = dataset.splits
    if options.split:
        if options.split not in splits_to_export:
            logger.warning(f"Split '{options.split}' not found in dataset")
            return
        splits_to_export = {options.split: dataset.splits[options.split]}

    # Export each split to FiftyOne
    # class_names from params (required for labeling)
    class_names = params.class_names or []
    if not class_names:
        logger.warning("No class names available in params, labels may not be displayed correctly")

    created_datasets = []
    for split_name, dataset_split in splits_to_export.items():
        if not dataset_split.images:
            logger.warning(f"Split '{split_name}' has no images, skipping")
            continue
        dataset_name = f"d20-{format_name}-{split_name}-{uuid4().hex}"
        logger.info(f"Exporting {format_name} split '{split_name}' to FiftyOne as '{dataset_name}'")

        # Create FiftyOne dataset from splits
        fo_dataset = _create_dataset_from_splits(format_name, dataset_split, class_names, dataset_name)

        # Ensure dataset is persisted
        fo_dataset.persistent = True

        # Verify dataset exists
        if fo_dataset.name in fo.list_datasets():  # type: ignore[attr-defined]
            created_datasets.append(dataset_name)
            logger.info(
                f"Dataset '{dataset_name}' created in FiftyOne with {len(dataset_split.images)} samples. "
                "Launch FiftyOne app to view: fiftyone app"
            )
        else:
            logger.error(f"Failed to create dataset '{dataset_name}' in FiftyOne")

    if created_datasets:
        logger.info(f"Created {len(created_datasets)} dataset(s) in FiftyOne: {', '.join(created_datasets)}")
        logger.info("To view datasets, run: fiftyone app")
        logger.info(f"Or open a specific dataset: fiftyone app --dataset {created_datasets[0]}")


def _create_dataset_from_splits(
    _dataset_format: str,
    dataset_split: Split,
    class_names: list[str],
    dataset_name: str,
) -> fo.Dataset:  # type:ignore[name-defined]
    """Create a FiftyOne dataset from Split object.

    Args:
        dataset_format: Format of the dataset
        dataset_split: Dataset split to convert
        class_names: List of class names for labeling
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
            logger.warning(f"Image file not found: {image_info.path}")
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

            label = class_names[ann.category_id] if ann.category_id < len(class_names) else f"class_{ann.category_id}"

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

    logger.info(f"Added {sample_count} samples to dataset '{dataset_name}' ({missing_images} images were missing)")
    if sample_count == 0:
        logger.error(f"No samples were added to dataset '{dataset_name}'. Check image paths.")
    return fo_dataset
