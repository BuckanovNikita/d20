"""Type definitions for dataset structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
class DatasetSplit:
    """A split of the dataset (train, val, test, etc.)."""

    name: str
    images: list[ImageInfo]
    annotations: list[Annotation]
