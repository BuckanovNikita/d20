"""Type definitions for dataset structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ImageInfo:
    """Information about an image in the dataset."""

    image_id: str
    file_name: str
    width: int
    height: int
    path: Path


@dataclass(frozen=True)
class Annotation:
    """Annotation for an object in an image."""

    image_id: str
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass
class Split:
    """A split of the dataset (train, val, test, etc.)."""

    name: str
    images: list[ImageInfo]
    annotations: list[Annotation]


@dataclass
class Dataset:
    """Universal dataset format containing all splits."""

    splits: dict[str, Split]
    class_names: list[str]
    _unique_image_sizes: set[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        """Compute unique image sizes after initialization."""
        if self._unique_image_sizes is None:
            sizes: set[tuple[int, int]] = set()
            for split in self.splits.values():
                for image in split.images:
                    sizes.add((image.width, image.height))
            object.__setattr__(self, "_unique_image_sizes", sizes)

    def get_unique_image_sizes(self) -> set[tuple[int, int]]:
        """Get unique image sizes (width, height) from all splits.

        Returns:
            Set of unique (width, height) tuples (precomputed)

        """
        return self._unique_image_sizes or set()

    def get_available_splits(self) -> list[str]:
        """Get list of available split names.

        Returns:
            List of split names

        """
        return sorted(self.splits.keys())

    def get_available_classes(self) -> list[str]:
        """Get unique class names from all annotations across all splits.

        Returns:
            List of unique class names (sorted by category ID)

        """
        class_ids: set[int] = set()
        for split in self.splits.values():
            for annotation in split.annotations:
                class_ids.add(annotation.category_id)

        # Map category IDs to class names and return sorted by ID
        return [
            self.class_names[category_id]
            for category_id in sorted(class_ids)
            if 0 <= category_id < len(self.class_names)
        ]


# Base protocol/interface for detected parameters
class DetectedParams(Protocol):
    """Interface for format-specific detected parameters."""

    input_path: Path | list[Path]
    class_names: list[str] | None
    splits: list[str] | None
    images_dir: str | None
    labels_dir: str | None
    annotations_dir: str | None


# Format-specific implementations
@dataclass
class YoloDetectedParams:
    """YOLO-specific detected parameters."""

    input_path: Path | list[Path]
    class_names: list[str] | None = None
    splits: list[str] | None = None
    images_dir: str | None = None
    labels_dir: str | None = None
    annotations_dir: str | None = None
    yaml_path: Path | None = None  # Format-specific: path to data.yaml
    dataset_root: Path | None = None  # Format-specific: root from YAML


@dataclass
class CocoDetectedParams:
    """COCO-specific detected parameters."""

    input_path: Path | list[Path]
    class_names: list[str] | None = None
    splits: list[str] | None = None
    images_dir: str | None = None
    labels_dir: str | None = None
    annotations_dir: str | None = None
    split_files: dict[str, Path] | None = None  # Format-specific: explicit split files


@dataclass
class VocDetectedParams:
    """VOC-specific detected parameters."""

    input_path: Path | list[Path]
    class_names: list[str] | None = None
    splits: list[str] | None = None
    images_dir: str | None = None
    labels_dir: str | None = None
    annotations_dir: str | None = None
    auto_detect_splits: bool = False  # Format-specific: auto-detect from ImageSets/Main


# Base protocol/interface for write parameters
class WriteDetectedParams(Protocol):
    """Interface for format-specific write parameters."""

    class_names: list[str]
    splits: list[str] | None
    images_dir: str
    labels_dir: str
    annotations_dir: str


@dataclass
class YoloWriteDetectedParams:
    """YOLO-specific write parameters."""

    class_names: list[str]
    splits: list[str] | None = None
    images_dir: str = "images"
    labels_dir: str = "labels"
    annotations_dir: str = "annotations"


@dataclass
class CocoWriteDetectedParams:
    """COCO-specific write parameters."""

    class_names: list[str]
    splits: list[str] | None = None
    images_dir: str = "images"
    labels_dir: str = "labels"
    annotations_dir: str = "annotations"


@dataclass
class VocWriteDetectedParams:
    """VOC-specific write parameters."""

    class_names: list[str]
    splits: list[str] | None = None
    images_dir: str = "images"
    labels_dir: str = "labels"
    annotations_dir: str = "annotations"


# Export options for FiftyOne
@dataclass
class ExportOptions:
    """Options for exporting to FiftyOne."""

    split: str | None = None  # Specific split to export (None = all splits)
