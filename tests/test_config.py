"""Tests for configuration management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from d20.config import ConversionConfig, load_config


def test_conversion_config_defaults() -> None:
    """Test ConversionConfig with default values."""
    config = ConversionConfig(class_names=["cat", "dog"])
    assert config.class_names == ["cat", "dog"]
    assert config.splits == ["data"]
    assert config.images_dir == "images"
    assert config.labels_dir == "labels"
    assert config.annotations_dir == "annotations"


def test_conversion_config_custom_values() -> None:
    """Test ConversionConfig with custom values."""
    config = ConversionConfig(
        class_names=["person", "bicycle"],
        splits=["train", "val"],
        images_dir="imgs",
        labels_dir="lbls",
        annotations_dir="anns",
    )
    assert config.class_names == ["person", "bicycle"]
    assert config.splits == ["train", "val"]
    assert config.images_dir == "imgs"
    assert config.labels_dir == "lbls"
    assert config.annotations_dir == "anns"


def test_conversion_config_split_names_property() -> None:
    """Test ConversionConfig.split_names property."""
    config = ConversionConfig(class_names=["cat"], splits=["train", "val"])
    assert config.split_names == ["train", "val"]


def test_conversion_config_split_names_default() -> None:
    """Test ConversionConfig.split_names property with default splits."""
    config = ConversionConfig(class_names=["cat"], splits=[])
    assert config.split_names == ["data"]


def test_conversion_config_validation_min_length() -> None:
    """Test ConversionConfig validation requires at least one class name."""
    with pytest.raises(ValidationError, match="class_names"):
        ConversionConfig(class_names=[])


def test_load_config_valid_yaml(tmp_path: Path) -> None:
    """Test load_config with valid YAML file."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "class_names": ["person", "bicycle", "car"],
        "splits": ["train", "val"],
        "images_dir": "images",
        "labels_dir": "labels",
        "annotations_dir": "annotations",
    }
    config_path.write_text(yaml.safe_dump(config_data))

    config = load_config(config_path)
    assert config.class_names == ["person", "bicycle", "car"]
    assert config.splits == ["train", "val"]
    assert config.images_dir == "images"
    assert config.labels_dir == "labels"
    assert config.annotations_dir == "annotations"


def test_load_config_minimal_yaml(tmp_path: Path) -> None:
    """Test load_config with minimal YAML file (only class_names)."""
    config_path = tmp_path / "config.yaml"
    config_data = {"class_names": ["cat", "dog"]}
    config_path.write_text(yaml.safe_dump(config_data))

    config = load_config(config_path)
    assert config.class_names == ["cat", "dog"]
    assert config.splits == ["data"]  # Default
    assert config.images_dir == "images"  # Default
    assert config.labels_dir == "labels"  # Default
    assert config.annotations_dir == "annotations"  # Default


def test_load_config_empty_yaml(tmp_path: Path) -> None:
    """Test load_config with empty YAML file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")

    with pytest.raises(ValidationError, match="class_names"):
        load_config(config_path)


def test_load_config_none_yaml(tmp_path: Path) -> None:
    """Test load_config with YAML file containing only null."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("null")

    with pytest.raises(ValidationError, match="class_names"):
        load_config(config_path)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Test load_config with invalid YAML file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("invalid: yaml: content: [unclosed")

    with pytest.raises(yaml.YAMLError):
        load_config(config_path)
