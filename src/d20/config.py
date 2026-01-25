from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ConversionConfig(BaseModel):
    class_names: list[str] = Field(min_length=1)
    splits: list[str] = Field(default_factory=lambda: ["data"])
    images_dir: str = "images"
    labels_dir: str = "labels"
    annotations_dir: str = "annotations"

    @property
    def split_names(self) -> list[str]:
        return self.splits or ["data"]


def load_config(path: Path) -> ConversionConfig:
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    return ConversionConfig.model_validate(data)
