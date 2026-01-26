"""Tests for dataset conversion functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from d20.convert import (
    ConversionRequest,
    UnsupportedInputFormatError,
    UnsupportedOutputFormatError,
    convert_dataset,
)
from d20.types import CocoWriteDetectedParams, YoloDetectedParams, YoloWriteDetectedParams


def test_unsupported_input_format_error() -> None:
    """Test UnsupportedInputFormatError exception."""
    error = UnsupportedInputFormatError("unknown_format")
    assert error.input_format == "unknown_format"
    assert str(error) == "Unsupported input format: unknown_format"


def test_unsupported_output_format_error() -> None:
    """Test UnsupportedOutputFormatError exception."""
    error = UnsupportedOutputFormatError("unknown_format")
    assert error.output_format == "unknown_format"
    assert str(error) == "Unsupported output format: unknown_format"


def test_convert_dataset_unsupported_input_format(tmp_path: Path) -> None:
    """Test convert_dataset() raises error for unsupported input format."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )
    write_params = YoloWriteDetectedParams(
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    request = ConversionRequest(
        input_format="unknown_format",
        output_format="yolo",
        input_path=tmp_path,
        output_dir=tmp_path / "output",
        read_params=params,
        write_params=write_params,
    )

    with pytest.raises(UnsupportedInputFormatError) as exc_info:
        convert_dataset(request)
    assert exc_info.value.input_format == "unknown_format"


def test_convert_dataset_unsupported_output_format(tmp_path: Path) -> None:
    """Test convert_dataset() raises error for unsupported output format."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )
    write_params = YoloWriteDetectedParams(
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    request = ConversionRequest(
        input_format="yolo",
        output_format="unknown_format",
        input_path=tmp_path,
        output_dir=tmp_path / "output",
        read_params=params,
        write_params=write_params,
    )

    with pytest.raises(UnsupportedOutputFormatError) as exc_info:
        convert_dataset(request)
    assert exc_info.value.output_format == "unknown_format"


def test_convert_dataset_success(tmp_path: Path) -> None:
    """Test convert_dataset() with valid formats."""
    # Create minimal dataset structure
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create a test image
    img_path = images_dir / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    # Create a label file
    label_path = labels_dir / "test.txt"
    label_path.write_text("0 0.5 0.5 0.1 0.1")  # Class 0, normalized bbox

    # Create YAML file
    yaml_file = dataset_dir / "data.yaml"
    yaml_data = {
        "path": ".",
        "names": ["cat"],
        "nc": 1,
        "data": "images",  # Use "data" split to match params
    }
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    params = YoloDetectedParams(
        input_path=dataset_dir,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
        yaml_path=yaml_file,
        dataset_root=dataset_dir,
    )
    write_params = CocoWriteDetectedParams(
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
        annotations_dir="annotations",
    )

    request = ConversionRequest(
        input_format="yolo",
        output_format="coco",
        input_path=dataset_dir,
        output_dir=tmp_path / "output",
        read_params=params,
        write_params=write_params,
    )

    # Should not raise
    convert_dataset(request)

    # Verify output was created
    output_dir = tmp_path / "output"
    assert output_dir.exists()
    assert (output_dir / "images").exists()
    assert (output_dir / "annotations").exists()


def test_convert_dataset_both_unsupported(tmp_path: Path) -> None:
    """Test convert_dataset() with both formats unsupported."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )
    write_params = YoloWriteDetectedParams(
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    request = ConversionRequest(
        input_format="unknown_input",
        output_format="unknown_output",
        input_path=tmp_path,
        output_dir=tmp_path / "output",
        read_params=params,
        write_params=write_params,
    )

    # Should raise input format error first
    with pytest.raises(UnsupportedInputFormatError):
        convert_dataset(request)
