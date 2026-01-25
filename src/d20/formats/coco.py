"""COCO format dataset reading and writing."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Any

from loguru import logger

from d20.types import Annotation, DatasetSplit, ImageInfo

if TYPE_CHECKING:
    from d20.config import ConversionConfig


class COCOCategoryMissingNameError(ValueError):
    """Raised when a COCO category is missing a name."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("COCO category missing name")


class UnknownClassNameError(ValueError):
    """Raised when an unknown class name is encountered."""

    def __init__(self, name: str) -> None:
        """Initialize error with class name."""
        super().__init__(f"Unknown class name: {name}")
        self.name = name


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def read_coco_dataset(input_dir: Path, config: ConversionConfig) -> list[DatasetSplit]:
    """Read a COCO format dataset from disk."""
    splits: list[DatasetSplit] = []
    for split in config.split_names:
        labels_path = input_dir / config.annotations_dir / f"{split}.json"
        if not labels_path.exists():
            logger.warning("COCO labels file missing: {}", labels_path)
            splits.append(DatasetSplit(name=split, images=[], annotations=[]))
            continue

        data = json.loads(labels_path.read_text())
        categories = data.get("categories", [])
        category_map: dict[int, int] = {}
        for category in categories:
            name = category.get("name")
            if not name:
                raise COCOCategoryMissingNameError
            if name not in config.class_names:
                raise UnknownClassNameError(name)
            category_map[int(category["id"])] = config.class_names.index(name)

        images_dir = _resolve_split_dir(input_dir / config.images_dir, split)
        images: list[ImageInfo] = []
        for image in data.get("images", []):
            file_name = image["file_name"]
            image_path = images_dir / file_name
            if not image_path.exists():
                image_path = input_dir / file_name
            images.append(
                ImageInfo(
                    image_id=str(image["id"]),
                    file_name=file_name,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    path=image_path,
                ),
            )

        annotations: list[Annotation] = []
        for annotation in data.get("annotations", []):
            category_id = category_map[int(annotation["category_id"])]
            bbox = annotation["bbox"]
            annotations.append(
                Annotation(
                    image_id=str(annotation["image_id"]),
                    category_id=category_id,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                ),
            )

        splits.append(DatasetSplit(name=split, images=images, annotations=annotations))

    return splits


def write_coco_dataset(output_dir: Path, config: ConversionConfig, splits: list[DatasetSplit]) -> None:
    """Write a dataset in COCO format to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / config.annotations_dir
    annotations_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"} for idx, name in enumerate(config.class_names)
    ]

    for split in splits:
        images_dir = output_dir / config.images_dir
        if split.name != "data":
            images_dir = images_dir / split.name
        images_dir.mkdir(parents=True, exist_ok=True)

        image_id_map: dict[str, int] = {}
        coco_images: list[dict[str, Any]] = []
        for idx, image in enumerate(split.images, start=1):
            image_id_map[image.image_id] = idx
            relative = Path(image.file_name)
            target_path = images_dir / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(image.path, target_path)

            coco_images.append(
                {
                    "id": idx,
                    "file_name": image.file_name,
                    "width": image.width,
                    "height": image.height,
                },
            )

        coco_annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for annotation in split.annotations:
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id_map[annotation.image_id],
                    "category_id": annotation.category_id + 1,
                    "bbox": list(annotation.bbox),
                    "area": float(annotation.bbox[2] * annotation.bbox[3]),
                    "iscrowd": 0,
                },
            )
            annotation_id += 1

        payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }
        labels_path = annotations_dir / f"{split.name}.json"
        labels_path.write_text(json.dumps(payload, ensure_ascii=True))
