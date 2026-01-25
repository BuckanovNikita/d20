from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageInfo:
    image_id: str
    file_name: str
    width: int
    height: int
    path: Path


@dataclass(frozen=True)
class Annotation:
    image_id: str
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass
class DatasetSplit:
    name: str
    images: list[ImageInfo]
    annotations: list[Annotation]
