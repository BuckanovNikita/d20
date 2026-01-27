"""Shared utilities for format converters."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from PIL import Image

from d20.types import ImageInfo

if TYPE_CHECKING:
    from collections.abc import Iterable

    from d20.types import Annotation

# Shared image extensions supported by all formats
IMAGE_EXTENSIONS = frozenset(
    {
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
)


def resolve_split_dir(base_dir: Path, split: str) -> Path:
    """Resolve split directory, falling back to base if split subdir doesn't exist.

    Args:
        base_dir: Base directory path
        split: Split name (e.g., 'train', 'val', 'test')

    Returns:
        Path to split directory if it exists, otherwise base directory

    """
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def iter_image_files(directory: Path, extensions: frozenset[str] | None = None) -> list[Path]:
    """Iterate over image files in directory recursively.

    Args:
        directory: Directory to search for images
        extensions: Set of image extensions to match. If None, uses IMAGE_EXTENSIONS.

    Returns:
        Sorted list of image file paths

    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    return sorted([path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions])


def group_annotations_by_image(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    """Group annotations by image_id.

    Args:
        annotations: Iterable of Annotation objects

    Returns:
        Dictionary mapping image_id to list of annotations for that image

    """
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def read_image_info(image_path: Path, images_dir: Path) -> ImageInfo:
    """Read image metadata and create ImageInfo.

    Args:
        image_path: Path to the image file
        images_dir: Base directory for images (used to compute relative path)

    Returns:
        ImageInfo object with image metadata

    """
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
