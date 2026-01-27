"""YOLO format dataset reading and writing."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Any, TypedDict

import yaml
from loguru import logger
from PIL import Image
from typing_extensions import Unpack, override

from d20.formats.base import FormatConverter, register_converter
from d20.types import (
    Annotation,
    Dataset,
    ImageInfo,
    Split,
    YoloDetectedParams,
    YoloWriteDetectedParams,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class _NormalizeOptions(TypedDict, total=False):
    """Options for normalizing input path."""

    yaml_path: Path | str


class UnknownClassIdError(ValueError):
    """Raised when an unknown class id is encountered."""

    def __init__(self, class_id: int) -> None:
        """Initialize error with class id."""
        super().__init__(f"Unknown class id: {class_id}")
        self.class_id = class_id


class EmptyPathListError(ValueError):
    """Raised when empty path list is provided."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("Empty path list provided")


class DatasetRootNotFoundError(ValueError):
    """Raised when dataset root path does not exist."""

    def __init__(self, root: Path) -> None:
        """Initialize error with root path."""
        super().__init__(f"Dataset root path does not exist: {root}")
        self.root = root


class MissingYoloNamesError(ValueError):
    """Raised when 'names' field is missing in YOLO YAML."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("Missing 'names' in YOLO YAML")


class UnsupportedYoloNamesStructureError(ValueError):
    """Raised when 'names' structure in YOLO YAML is unsupported."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("Unsupported 'names' structure in YOLO YAML")


class InvalidYoloCoordinateError(ValueError):
    """Raised when YOLO coordinates are out of valid range [0, 1]."""

    def __init__(self, coord_name: str, value: float) -> None:
        """Initialize error with coordinate name and value."""
        super().__init__(f"Invalid YOLO coordinate {coord_name}: {value} (must be in [0, 1])")
        self.coord_name = coord_name
        self.value = value


class InvalidYoloNumericValueError(ValueError):
    """Raised when YOLO annotation contains invalid numeric value."""

    def __init__(self, line: str, field: str, value: str, error: Exception) -> None:
        """Initialize error with line, field, value and original error."""
        super().__init__(
            f"Invalid numeric value in YOLO annotation line '{line}': field '{field}' = '{value}': {error}"
        )
        self.line = line
        self.field = field
        self.value = value
        self.original_error = error


class InvalidYoloBboxError(ValueError):
    """Raised when calculated bounding box is invalid."""

    def __init__(self, bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> None:
        """Initialize error with bbox and image size."""
        x, y, w, h = bbox
        img_w, img_h = image_size
        super().__init__(
            f"Invalid bounding box ({x}, {y}, {w}, {h}) for image size ({img_w}, {img_h}): "
            f"width={w} and height={h} must be positive, "
            f"bbox must be within image bounds",
        )
        self.bbox = bbox
        self.image_size = image_size


IMAGE_EXTS = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _iter_images(images_dir: Path) -> list[Path]:
    return sorted([path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS])


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def _read_image_info(image_path: Path, images_dir: Path) -> ImageInfo:
    relative = image_path.relative_to(images_dir)
    with Image.open(image_path) as image:
        width, height = image.size

    return ImageInfo(
        image_id=image_path.stem,
        file_name=relative.as_posix(),
        width=width,
        height=height,
        path=image_path,
    )


def _group_annotations(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def yaml_safe_dump(data: dict[str, Any]) -> str:
    """Safely dump data to YAML string."""
    return yaml.safe_dump(data, sort_keys=False)


@register_converter("yolo")
class YoloConverter(FormatConverter):
    """Converter for YOLO format datasets."""

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "yolo"

    def _normalize_input_path(
        self,
        input_path: Path | list[Path],
        **options: Unpack[_NormalizeOptions],
    ) -> tuple[Path, dict[str, Any]]:
        """Normalize input path and determine data type.

        Args:
            input_path: Input path (directory, YAML file, or list)
            **options: Additional options

        Returns:
            Tuple of (normalized path, updated options)

        """
        # If list is provided, use first path (YOLO doesn't support list yet)
        if isinstance(input_path, list):
            if len(input_path) > 0:
                input_path = input_path[0]
            else:
                raise EmptyPathListError

        path = Path(input_path)

        # Check explicit yaml_path option
        if options.get("yaml_path"):
            yaml_path_value = options["yaml_path"]
            yaml_path = yaml_path_value if isinstance(yaml_path_value, Path) else Path(yaml_path_value)
            if yaml_path.exists():
                return yaml_path, dict(options)

        # Auto-detect YAML file
        if path.suffix.lower() in (".yaml", ".yml") and path.exists():
            return path, dict(options)

        # Look for data.yaml in directory
        if path.is_dir():
            yaml_path = path / "data.yaml"
            if yaml_path.exists():
                return yaml_path, dict(options)

        # Return as-is (directory)
        return path, dict(options)

    def _read_from_yaml(self, yaml_path: Path) -> tuple[Path, dict[str, Any]]:
        """Read YOLO YAML configuration file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Tuple of (dataset root path, YAML data)

        """
        data = yaml.safe_load(yaml_path.read_text()) or {}

        # Resolve root path
        root_value = data.get("path", ".")
        if root_value:
            root = Path(root_value)
            if not root.is_absolute():
                root = (yaml_path.parent / root).resolve()
        else:
            root = yaml_path.parent

        if not root.exists():
            raise DatasetRootNotFoundError(root)

        return root, data

    def _load_class_names_from_yaml(self, data: dict[str, Any]) -> list[str]:
        """Load class names from YAML data.

        Args:
            data: YAML data dictionary

        Returns:
            List of class names

        """
        names = data.get("names")
        if not names:
            raise MissingYoloNamesError

        if isinstance(names, list):
            return names

        if isinstance(names, dict):

            def _key_as_int(key: str | int) -> int:
                return int(key)

            return [names[key] for key in sorted(names.keys(), key=_key_as_int)]

        raise UnsupportedYoloNamesStructureError

    def _detect_splits_from_yaml(self, data: dict[str, Any]) -> list[str]:
        """Detect splits from YAML data.

        Args:
            data: YAML data dictionary

        Returns:
            List of split names

        """
        splits = [split_name for split_name in ("train", "val", "test") if data.get(split_name)]
        return splits if splits else ["data"]

    def _determine_input_config(
        self,
        params: YoloDetectedParams,
    ) -> tuple[Path, list[str], list[str]]:
        """Determine input directory, class names, and splits from params.

        Args:
            params: YoloDetectedParams with all necessary information

        Returns:
            Tuple of (input_dir, class_names, splits)

        """
        if params.yaml_path and params.yaml_path.exists():
            dataset_root, yaml_data = self._read_from_yaml(params.yaml_path)
            class_names = self._load_class_names_from_yaml(yaml_data)
            splits = self._detect_splits_from_yaml(yaml_data)
            input_dir = dataset_root
        elif params.dataset_root:
            input_dir = params.dataset_root
            class_names = params.class_names or []
            splits = params.splits or ["data"]
        else:
            # Fallback to input_path
            input_dir = Path(params.input_path[0]) if isinstance(params.input_path, list) else Path(params.input_path)
            class_names = params.class_names or []
            splits = params.splits or ["data"]

        if not class_names:
            raise MissingYoloNamesError

        return input_dir, class_names, splits

    def _parse_yolo_annotation_line(
        self,
        line: str,
        image: ImageInfo,
        class_names: list[str],
    ) -> Annotation | None:
        """Parse a single line from a YOLO label file.

        Args:
            line: Line from label file
            image: ImageInfo for the image
            class_names: List of class names

        Returns:
            Annotation object or None if line is invalid

        Raises:
            InvalidYoloNumericValueError: If numeric conversion fails
            UnknownClassIdError: If class_id is out of range
            InvalidYoloCoordinateError: If coordinates are out of [0, 1] range
            InvalidYoloBboxError: If calculated bbox is invalid

        """
        stripped = line.strip()
        if not stripped:
            return None

        parts = stripped.split()
        if len(parts) != 5:
            logger.warning(f"Skipping invalid YOLO label line: {line}")
            return None

        # Validate and convert class_id
        try:
            class_id = int(parts[0])
        except ValueError as e:
            raise InvalidYoloNumericValueError(line, "class_id", parts[0], e) from e

        if class_id < 0:
            raise UnknownClassIdError(class_id)
        if class_id >= len(class_names):
            raise UnknownClassIdError(class_id)

        # Validate and convert coordinates
        try:
            x_center = float(parts[1])
        except ValueError as e:
            raise InvalidYoloNumericValueError(line, "x_center", parts[1], e) from e

        try:
            y_center = float(parts[2])
        except ValueError as e:
            raise InvalidYoloNumericValueError(line, "y_center", parts[2], e) from e

        try:
            width = float(parts[3])
        except ValueError as e:
            raise InvalidYoloNumericValueError(line, "width", parts[3], e) from e

        try:
            height = float(parts[4])
        except ValueError as e:
            raise InvalidYoloNumericValueError(line, "height", parts[4], e) from e

        # Validate coordinate ranges [0, 1]
        if not (0.0 <= x_center <= 1.0):
            raise InvalidYoloCoordinateError("x_center", x_center)
        if not (0.0 <= y_center <= 1.0):
            raise InvalidYoloCoordinateError("y_center", y_center)
        if not (0.0 <= width <= 1.0):
            raise InvalidYoloCoordinateError("width", width)
        if not (0.0 <= height <= 1.0):
            raise InvalidYoloCoordinateError("height", height)

        # Calculate bounding box
        bbox_x = (x_center - width / 2.0) * image.width
        bbox_y = (y_center - height / 2.0) * image.height
        bbox_w = width * image.width
        bbox_h = height * image.height

        # Validate calculated bbox
        if bbox_w <= 0 or bbox_h <= 0:
            raise InvalidYoloBboxError((bbox_x, bbox_y, bbox_w, bbox_h), (image.width, image.height))

        # Check if bbox is within image bounds (with small tolerance for floating point)
        if (
            bbox_x < -0.5
            or bbox_y < -0.5
            or (bbox_x + bbox_w) > image.width + 0.5
            or (bbox_y + bbox_h) > image.height + 0.5
        ):
            raise InvalidYoloBboxError((bbox_x, bbox_y, bbox_w, bbox_h), (image.width, image.height))

        return Annotation(
            image_id=image.image_id,
            category_id=class_id,
            bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
        )

    def _read_split(
        self,
        split_name: str,
        input_dir: Path,
        images_dir_name: str,
        labels_dir_name: str,
        class_names: list[str],
    ) -> Split:
        """Read a single split from YOLO format.

        Args:
            split_name: Name of the split
            input_dir: Root directory of the dataset
            images_dir_name: Name of images directory
            labels_dir_name: Name of labels directory
            class_names: List of class names

        Returns:
            Split object with images and annotations

        """
        images_dir = _resolve_split_dir(input_dir / images_dir_name, split_name)
        labels_dir = _resolve_split_dir(input_dir / labels_dir_name, split_name)

        if not images_dir.exists():
            logger.warning(f"YOLO images directory missing: {images_dir}")
            return Split(name=split_name, images=[], annotations=[])

        images = [_read_image_info(path, images_dir) for path in _iter_images(images_dir)]
        annotations: list[Annotation] = []

        for image in images:
            label_path = labels_dir / Path(image.file_name).with_suffix(".txt")
            if not label_path.exists():
                continue

            for line in label_path.read_text().splitlines():
                annotation = self._parse_yolo_annotation_line(line, image, class_names)
                if annotation:
                    annotations.append(annotation)

        return Split(name=split_name, images=images, annotations=annotations)

    def autodetect(
        self,
        input_path: Path | list[Path],
    ) -> YoloDetectedParams:
        """Autodetect YOLO dataset parameters without reading the full dataset.

        Args:
            input_path: Path to dataset directory or YAML file

        Returns:
            YoloDetectedParams with detected parameters

        """
        # Normalize input path
        if isinstance(input_path, list):
            if len(input_path) > 0:
                input_path = input_path[0]
            else:
                raise EmptyPathListError

        path = Path(input_path)
        yaml_path: Path | None = None
        dataset_root: Path | None = None
        class_names: list[str] | None = None
        splits: list[str] | None = None

        # Check if it's a YAML file
        if path.suffix.lower() in (".yaml", ".yml") and path.exists():
            yaml_path = path
            dataset_root, yaml_data = self._read_from_yaml(path)
            try:
                class_names = self._load_class_names_from_yaml(yaml_data)
                splits = self._detect_splits_from_yaml(yaml_data)
            except (MissingYoloNamesError, UnsupportedYoloNamesStructureError) as e:
                # If YAML doesn't have names, we can't detect class_names
                # Log warning but continue autodetect (class_names will be None)
                logger.warning(f"Could not detect class names from YAML during autodetect: {e}")
        elif path.is_dir():
            # Look for data.yaml in directory
            potential_yaml = path / "data.yaml"
            if potential_yaml.exists():
                yaml_path = potential_yaml
                dataset_root, yaml_data = self._read_from_yaml(potential_yaml)
                try:
                    class_names = self._load_class_names_from_yaml(yaml_data)
                    splits = self._detect_splits_from_yaml(yaml_data)
                except (MissingYoloNamesError, UnsupportedYoloNamesStructureError) as e:
                    # If YAML doesn't have names, we can't detect class_names
                    # Log warning but continue autodetect (class_names will be None)
                    logger.warning(f"Could not detect class names from YAML during autodetect: {e}")
            else:
                dataset_root = path

        # Default directory structure detection
        images_dir: str | None = "images"
        labels_dir: str | None = "labels"
        annotations_dir: str | None = None  # YOLO doesn't use annotations_dir

        return YoloDetectedParams(
            input_path=input_path,
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=annotations_dir,
            yaml_path=yaml_path,
            dataset_root=dataset_root,
        )

    @override
    def read(
        self,
        params: YoloDetectedParams,  # type: ignore[override]
    ) -> Dataset:
        """Read a YOLO format dataset from disk.

        Args:
            params: YoloDetectedParams with all necessary information

        Returns:
            Dataset containing all splits

        """
        input_dir, class_names, splits = self._determine_input_config(params)

        images_dir_name = params.images_dir or "images"
        labels_dir_name = params.labels_dir or "labels"

        splits_dict: dict[str, Split] = {}
        for split_name in splits:
            splits_dict[split_name] = self._read_split(
                split_name,
                input_dir,
                images_dir_name,
                labels_dir_name,
                class_names,
            )

        return Dataset(splits=splits_dict, class_names=class_names)

    @override
    def write(
        self,
        output_dir: Path,
        dataset: Dataset,
        params: YoloWriteDetectedParams,  # type: ignore[override]
    ) -> None:
        """Write a dataset in YOLO format to disk.

        Args:
            output_dir: Output directory path
            dataset: Dataset containing all splits
            params: YoloWriteDetectedParams with write configuration

        """
        output_dir.mkdir(parents=True, exist_ok=True)

        splits_to_write = params.splits or list(dataset.splits.keys())
        for split_name in splits_to_write:
            if split_name not in dataset.splits:
                logger.warning(f"Split '{split_name}' not found in dataset, skipping")
                continue

            split = dataset.splits[split_name]
            images_dir = output_dir / params.images_dir
            labels_dir = output_dir / params.labels_dir
            if split.name != "data":
                images_dir = images_dir / split.name
                labels_dir = labels_dir / split.name

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            grouped = _group_annotations(split.annotations)
            for image in split.images:
                relative = Path(image.file_name)
                target_image_path = images_dir / relative
                target_image_path.parent.mkdir(parents=True, exist_ok=True)
                copy2(image.path, target_image_path)

                label_path = labels_dir / relative.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for annotation in grouped.get(image.image_id, []):
                    x, y, w, h = annotation.bbox
                    x_center = (x + w / 2.0) / image.width
                    y_center = (y + h / 2.0) / image.height
                    w_norm = w / image.width
                    h_norm = h / image.height
                    lines.append(f"{annotation.category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                label_path.write_text("\n".join(lines))

        self._write_data_yaml(output_dir, params)

    def _write_data_yaml(self, output_dir: Path, params: YoloWriteDetectedParams) -> None:
        """Write data.yaml file."""
        data = {
            "path": ".",
            "names": params.class_names,
            "nc": len(params.class_names),
        }
        splits_to_write = params.splits or ["data"]
        for split in splits_to_write:
            if split == "data":
                data[split] = f"{params.images_dir}"
            else:
                data[split] = f"{params.images_dir}/{split}"

        (output_dir / "data.yaml").write_text(
            yaml_safe_dump(data),
        )
