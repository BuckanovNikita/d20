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
    """Load conversion configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        ConversionConfig instance

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If config validation fails

    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        msg = f"Failed to parse YAML configuration file {path}: {e}"
        raise ValueError(msg) from e
    except OSError as e:
        msg = f"Failed to read configuration file {path}: {e}"
        raise ValueError(msg) from e

    if data is None:
        data = {}

    try:
        return ConversionConfig.model_validate(data)
    except Exception as e:
        msg = f"Configuration validation failed for {path}: {e}"
        raise ValueError(msg) from e
