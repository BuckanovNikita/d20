"""Configuration management for dataset conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class ConversionConfig(BaseModel):
    """Configuration for dataset conversion."""

    class_names: list[str] = Field(min_length=1)
    splits: list[str] = Field(default_factory=lambda: ["data"])
    images_dir: str = "images"
    labels_dir: str = "labels"
    annotations_dir: str = "annotations"

    @property
    def split_names(self) -> list[str]:
        """Return split names, defaulting to ['data'] if empty."""
        return self.splits or ["data"]


def load_config(path: Path) -> ConversionConfig:
    """Load conversion configuration from a YAML file."""
    data = yaml.safe_load(path.read_text())
    if data is None:
        data = {}
    return ConversionConfig.model_validate(data)
