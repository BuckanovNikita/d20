"""COCO format dataset reading and writing."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Any

from loguru import logger

from d20.formats.base import FormatConverter, register_converter
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


class SplitNamesMismatchError(ValueError):
    """Raised when number of split names doesn't match number of files."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("Number of split names must match number of JSON files")


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


@register_converter("coco")
class CocoConverter(FormatConverter):
    """Converter for COCO format datasets."""

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "coco"

    def _read_split_from_json(
        self,
        json_path: Path,
        input_dir: Path,
        config: ConversionConfig,
        split_name: str,
    ) -> DatasetSplit:
        """Read a single split from a COCO JSON file.

        Args:
            json_path: Path to COCO JSON file
            input_dir: Base input directory
            config: Conversion configuration
            split_name: Name of the split

        Returns:
            Dataset split

        """
        if not json_path.exists():
            logger.warning("COCO labels file missing: {}", json_path)
            return DatasetSplit(name=split_name, images=[], annotations=[])

        data = json.loads(json_path.read_text())
        categories = data.get("categories", [])
        category_map: dict[int, int] = {}
        for category in categories:
            name = category.get("name")
            if not name:
                raise COCOCategoryMissingNameError
            if name not in config.class_names:
                raise UnknownClassNameError(name)
            category_map[int(category["id"])] = config.class_names.index(name)

        images_dir = _resolve_split_dir(input_dir / config.images_dir, split_name)
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

        return DatasetSplit(name=split_name, images=images, annotations=annotations)

    def read(
        self,
        input_path: Path | list[Path],
        config: ConversionConfig,
        **options: Any,
    ) -> list[DatasetSplit]:
        """Read a COCO format dataset from disk.

        Args:
            input_path: Path to dataset directory or list of JSON file paths
            config: Conversion configuration
            **options: Additional options:
                - split_files: dict[str, Path] for explicit file mapping
                - split_names: list[str] for split names when using list[Path]

        Returns:
            List of dataset splits

        """
        splits: list[DatasetSplit] = []

        # Handle list of paths (explicit JSON files)
        if isinstance(input_path, list):
            split_names = options.get("split_names", [])
            if not split_names:
                # Generate split names from file names
                split_names = [Path(p).stem for p in input_path]

            if len(split_names) != len(input_path):
                raise SplitNamesMismatchError

            # Determine base directory (parent of first file)
            base_dir = input_path[0].parent if input_path else Path.cwd()

            for json_path, split_name in zip(input_path, split_names, strict=True):
                split_data = self._read_split_from_json(Path(json_path), base_dir, config, split_name)
                splits.append(split_data)

            return splits

        # Handle split_files option
        if options.get("split_files"):
            split_files: dict[str, Path] = options["split_files"]
            input_dir = Path(input_path)
            for split_name, json_path in split_files.items():
                split = self._read_split_from_json(Path(json_path), input_dir, config, split_name)
                splits.append(split)
            return splits

        # Standard directory-based approach
        input_dir = Path(input_path)
        for split_name in config.split_names:
            labels_path = input_dir / config.annotations_dir / f"{split_name}.json"
            split_data = self._read_split_from_json(labels_path, input_dir, config, split_name)
            splits.append(split_data)

        return splits

    def write(
        self,
        output_dir: Path,
        config: ConversionConfig,
        splits: list[DatasetSplit],
        **_options: Any,
    ) -> None:
        """Write a dataset in COCO format to disk.

        Args:
            output_dir: Output directory path
            config: Conversion configuration
            splits: List of dataset splits to write
            **options: Additional options (unused)

        """
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
