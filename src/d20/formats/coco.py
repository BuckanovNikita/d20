"""COCO format dataset reading and writing."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import Any

from loguru import logger
from typing_extensions import override

from d20.formats.base import FormatConverter, register_converter
from d20.types import (
    Annotation,
    CocoDetectedParams,
    CocoWriteDetectedParams,
    Dataset,
    ImageInfo,
    Split,
)


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


class InvalidCocoJsonError(ValueError):
    """Raised when COCO JSON file is invalid or cannot be parsed."""

    def __init__(self, json_path: Path, error: Exception) -> None:
        """Initialize error with JSON path and original error."""
        super().__init__(f"Failed to parse COCO JSON file {json_path}: {error}")
        self.json_path = json_path
        self.original_error = error


class InvalidCocoCategoryIdError(ValueError):
    """Raised when annotation references non-existent category_id."""

    def __init__(self, category_id: int, available_ids: list[int]) -> None:
        """Initialize error with category_id and available IDs."""
        super().__init__(f"Invalid category_id {category_id}, available IDs: {available_ids}")
        self.category_id = category_id
        self.available_ids = available_ids


class InvalidCocoImageIdError(ValueError):
    """Raised when annotation references non-existent image_id."""

    def __init__(self, image_id: str, available_ids: list[str]) -> None:
        """Initialize error with image_id and available IDs."""
        super().__init__(f"Invalid image_id {image_id}, available IDs: {list(available_ids)[:10]}...")
        self.image_id = image_id
        self.available_ids = available_ids


class InvalidCocoBboxError(ValueError):
    """Raised when COCO bbox is invalid."""

    def __init__(self, bbox: list[float], image_size: tuple[int, int] | None = None) -> None:
        """Initialize error with bbox and optional image size."""
        if image_size:
            img_w, img_h = image_size
            msg = (
                f"Invalid bounding box {bbox} for image size ({img_w}, {img_h}): "
                f"bbox must have 4 elements [x, y, width, height], all non-negative, "
                f"and within image bounds"
            )
        else:
            msg = f"Invalid bounding box {bbox}: bbox must have 4 elements [x, y, width, height], all non-negative"
        super().__init__(msg)
        self.bbox = bbox
        self.image_size = image_size


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def _extract_categories_from_json_file(json_path: Path) -> tuple[set[str], dict[str, int]]:
    """Extract category names and mapping from a COCO JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (set of category names, dict mapping name to id)

    Raises:
        InvalidCocoJsonError: If JSON file cannot be parsed or read

    """
    category_names: set[str] = set()
    category_map: dict[str, int] = {}
    try:
        data = json.loads(json_path.read_text())
        categories = data.get("categories", [])
        for category in categories:
            name = category.get("name")
            cat_id = category.get("id")
            if name:
                category_names.add(name)
                if cat_id is not None and name not in category_map:
                    category_map[name] = int(cat_id)
    except json.JSONDecodeError as e:
        raise InvalidCocoJsonError(json_path, e) from e
    except OSError as e:
        raise InvalidCocoJsonError(json_path, e) from e
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidCocoJsonError(json_path, e) from e
    return category_names, category_map


@register_converter("coco")
class CocoConverter(FormatConverter):
    """Converter for COCO format datasets."""

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "coco"

    def autodetect(
        self,
        input_path: Path | list[Path],
    ) -> CocoDetectedParams:
        """Autodetect COCO dataset parameters without reading the full dataset.

        Args:
            input_path: Path to dataset directory or list of JSON file paths

        Returns:
            CocoDetectedParams with detected parameters

        """
        class_names: list[str] | None = None
        splits: list[str] | None = None
        split_files: dict[str, Path] | None = None

        # Handle list of paths (explicit JSON files)
        if isinstance(input_path, list):
            split_files = {}
            all_category_names: set[str] = set()
            category_map: dict[str, int] = {}

            for json_path in input_path:
                path = Path(json_path)
                split_name = path.stem
                split_files[split_name] = path

                # Extract categories from JSON
                names, mapping = _extract_categories_from_json_file(path)
                all_category_names.update(names)
                category_map.update(mapping)

            if all_category_names:
                if category_map:
                    class_names = sorted(all_category_names, key=lambda n: category_map.get(n, 999))
                else:
                    class_names = sorted(all_category_names)
            splits = list(split_files.keys())
        else:
            # Directory-based approach
            input_dir = Path(input_path)
            annotations_dir = "annotations"

            # Detect splits from JSON files in annotations directory
            annotations_path = input_dir / annotations_dir
            if annotations_path.exists():
                json_files = list(annotations_path.glob("*.json"))
                if json_files:
                    splits = [f.stem for f in json_files]
                    dir_category_names: set[str] = set()
                    dir_category_map: dict[str, int] = {}

                    # Extract categories from all JSON files
                    for json_path in json_files:
                        names, mapping = _extract_categories_from_json_file(json_path)
                        dir_category_names.update(names)
                        dir_category_map.update(mapping)

                    if dir_category_names:
                        if dir_category_map:
                            class_names = sorted(dir_category_names, key=lambda n: dir_category_map.get(n, 999))
                        else:
                            class_names = sorted(dir_category_names)

        images_dir: str | None = "images"
        labels_dir: str | None = None  # COCO doesn't use labels_dir
        final_annotations_dir: str | None = "annotations"

        return CocoDetectedParams(
            input_path=input_path,
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=final_annotations_dir,
            split_files=split_files,
        )

    def _read_split_from_json(
        self,
        json_path: Path,
        input_dir: Path,
        class_names: list[str],
        images_dir_name: str,
        split_name: str,
    ) -> Split:
        """Read a single split from a COCO JSON file.

        Args:
            json_path: Path to COCO JSON file
            input_dir: Base input directory
            class_names: List of class names
            images_dir_name: Name of images directory
            split_name: Name of the split

        Returns:
            Dataset split

        """
        if not json_path.exists():
            logger.warning(f"COCO labels file missing: {json_path}")
            return Split(name=split_name, images=[], annotations=[])

        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise InvalidCocoJsonError(json_path, e) from e

        categories = data.get("categories", [])
        category_map: dict[int, int] = {}
        for category in categories:
            name = category.get("name")
            if not name:
                raise COCOCategoryMissingNameError
            if name not in class_names:
                raise UnknownClassNameError(name)
            category_map[int(category["id"])] = class_names.index(name)

        images_dir = _resolve_split_dir(input_dir / images_dir_name, split_name)
        images: list[ImageInfo] = []
        image_id_set: set[str] = set()
        for image in data.get("images", []):
            try:
                file_name = image["file_name"]
                image_id = str(image["id"])
                width = int(image["width"])
                height = int(image["height"])
            except (KeyError, ValueError, TypeError) as e:
                raise InvalidCocoJsonError(json_path, ValueError(f"Invalid image data: {e}")) from e

            if width <= 0 or height <= 0:
                raise InvalidCocoBboxError([0, 0, width, height], (width, height))

            image_path = images_dir / file_name
            if not image_path.exists():
                image_path = input_dir / file_name
            images.append(
                ImageInfo(
                    image_id=image_id,
                    file_name=file_name,
                    width=width,
                    height=height,
                    path=image_path,
                ),
            )
            image_id_set.add(image_id)

        annotations: list[Annotation] = []
        for annotation in data.get("annotations", []):
            try:
                annotation_category_id = int(annotation["category_id"])
                annotation_image_id = str(annotation["image_id"])
                bbox = annotation["bbox"]
            except (KeyError, ValueError, TypeError) as e:
                raise InvalidCocoJsonError(json_path, ValueError(f"Invalid annotation data: {e}")) from e

            # Validate category_id exists in category_map
            if annotation_category_id not in category_map:
                available_ids = list(category_map.keys())
                raise InvalidCocoCategoryIdError(annotation_category_id, available_ids)
            category_id = category_map[annotation_category_id]

            # Validate image_id exists
            if annotation_image_id not in image_id_set:
                raise InvalidCocoImageIdError(annotation_image_id, image_id_set)

            # Validate bbox
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise InvalidCocoBboxError(bbox)

            try:
                bbox_x = float(bbox[0])
                bbox_y = float(bbox[1])
                bbox_w = float(bbox[2])
                bbox_h = float(bbox[3])
            except (ValueError, TypeError) as e:
                raise InvalidCocoBboxError(bbox) from e

            # Find corresponding image for size validation
            image_info = next((img for img in images if img.image_id == annotation_image_id), None)
            if image_info:
                if bbox_w <= 0 or bbox_h <= 0:
                    raise InvalidCocoBboxError(bbox, (image_info.width, image_info.height))
                # Check if bbox is within image bounds (with small tolerance for floating point)
                if (
                    bbox_x < -0.5
                    or bbox_y < -0.5
                    or (bbox_x + bbox_w) > image_info.width + 0.5
                    or (bbox_y + bbox_h) > image_info.height + 0.5
                ):
                    raise InvalidCocoBboxError(bbox, (image_info.width, image_info.height))

            annotations.append(
                Annotation(
                    image_id=annotation_image_id,
                    category_id=category_id,
                    bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                ),
            )

        return Split(name=split_name, images=images, annotations=annotations)

    @override
    def read(
        self,
        params: CocoDetectedParams,  # type: ignore[override]
    ) -> Dataset:
        """Read a COCO format dataset from disk.

        Args:
            params: CocoDetectedParams with all necessary information

        Returns:
            Dataset containing all splits

        """
        if not params.class_names:
            msg = "Class names are required for COCO format"
            raise ValueError(msg)

        class_names = params.class_names
        images_dir_name = params.images_dir or "images"
        splits_dict: dict[str, Split] = {}

        # Handle split_files option
        if params.split_files:
            if isinstance(params.input_path, list):
                base_dir = params.input_path[0].parent if params.input_path else Path.cwd()
            else:
                base_dir = Path(params.input_path)

            for split_name, json_path in params.split_files.items():
                split = self._read_split_from_json(
                    json_path,
                    base_dir,
                    class_names,
                    images_dir_name,
                    split_name,
                )
                splits_dict[split_name] = split
        elif isinstance(params.input_path, list):
            # List of JSON files
            base_dir = params.input_path[0].parent if params.input_path else Path.cwd()
            split_names = params.splits or [Path(p).stem for p in params.input_path]

            if len(split_names) != len(params.input_path):
                raise SplitNamesMismatchError

            for json_path, split_name in zip(params.input_path, split_names, strict=True):
                split = self._read_split_from_json(
                    Path(json_path),
                    base_dir,
                    class_names,
                    images_dir_name,
                    split_name,
                )
                splits_dict[split_name] = split
        else:
            # Standard directory-based approach
            input_dir = Path(params.input_path)
            annotations_dir = params.annotations_dir or "annotations"
            splits = params.splits or ["data"]

            for split_name in splits:
                labels_path = input_dir / annotations_dir / f"{split_name}.json"
                split = self._read_split_from_json(
                    labels_path,
                    input_dir,
                    class_names,
                    images_dir_name,
                    split_name,
                )
                splits_dict[split_name] = split

        return Dataset(splits=splits_dict, class_names=class_names)

    @override
    def write(
        self,
        output_dir: Path,
        dataset: Dataset,
        params: CocoWriteDetectedParams,  # type: ignore[override]
    ) -> None:
        """Write a dataset in COCO format to disk.

        Args:
            output_dir: Output directory path
            dataset: Dataset containing all splits
            params: CocoWriteDetectedParams with write configuration

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = output_dir / params.annotations_dir
        annotations_dir.mkdir(parents=True, exist_ok=True)

        categories = [
            {"id": idx + 1, "name": name, "supercategory": "object"} for idx, name in enumerate(params.class_names)
        ]

        splits_to_write = params.splits or list(dataset.splits.keys())
        for split_name in splits_to_write:
            if split_name not in dataset.splits:
                logger.warning(f"Split '{split_name}' not found in dataset, skipping")
                continue

            split = dataset.splits[split_name]
            images_dir = output_dir / params.images_dir
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
                if annotation.image_id not in image_id_map:
                    raise InvalidCocoImageIdError(annotation.image_id, list(image_id_map.keys()))
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
