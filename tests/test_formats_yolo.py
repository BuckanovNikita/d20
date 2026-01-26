"""Tests for YOLO format converter."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from d20.formats.yolo import (
    DatasetRootNotFoundError,
    EmptyPathListError,
    MissingYoloNamesError,
    UnknownClassIdError,
    UnsupportedYoloNamesStructureError,
    YoloConverter,
    yaml_safe_dump,
)
from d20.types import YoloDetectedParams, YoloWriteDetectedParams


def test_unknown_class_id_error() -> None:
    """Test UnknownClassIdError exception."""
    error = UnknownClassIdError(5)
    assert error.class_id == 5
    assert str(error) == "Unknown class id: 5"


def test_empty_path_list_error() -> None:
    """Test EmptyPathListError exception."""
    error = EmptyPathListError()
    assert str(error) == "Empty path list provided"


def test_dataset_root_not_found_error() -> None:
    """Test DatasetRootNotFoundError exception."""
    root = Path("/nonexistent/path")
    error = DatasetRootNotFoundError(root)
    assert error.root == root
    assert str(error) == f"Dataset root path does not exist: {root}"


def test_missing_yolo_names_error() -> None:
    """Test MissingYoloNamesError exception."""
    error = MissingYoloNamesError()
    assert str(error) == "Missing 'names' in YOLO YAML"


def test_unsupported_yolo_names_structure_error() -> None:
    """Test UnsupportedYoloNamesStructureError exception."""
    error = UnsupportedYoloNamesStructureError()
    assert str(error) == "Unsupported 'names' structure in YOLO YAML"


def test_yaml_safe_dump() -> None:
    """Test yaml_safe_dump() function."""
    data = {"path": ".", "names": ["cat", "dog"], "nc": 2}
    result = yaml_safe_dump(data)
    assert isinstance(result, str)
    parsed = yaml.safe_load(result)
    assert parsed == data


def test_yolo_converter_format_name() -> None:
    """Test YoloConverter.format_name property."""
    converter = YoloConverter()
    assert converter.format_name == "yolo"


def test_yolo_converter_normalize_input_path_directory(tmp_path: Path) -> None:
    """Test _normalize_input_path() with directory."""
    converter = YoloConverter()
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()

    path, options = converter._normalize_input_path(input_dir)
    assert path == input_dir
    assert isinstance(options, dict)


def test_yolo_converter_normalize_input_path_yaml_file(tmp_path: Path) -> None:
    """Test _normalize_input_path() with YAML file."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names: [cat, dog]")

    path, _options = converter._normalize_input_path(yaml_file)
    assert path == yaml_file


def test_yolo_converter_normalize_input_path_list(tmp_path: Path) -> None:
    """Test _normalize_input_path() with list."""
    converter = YoloConverter()
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()

    path, _options = converter._normalize_input_path([input_dir])
    assert path == input_dir


def test_yolo_converter_normalize_input_path_empty_list() -> None:
    """Test _normalize_input_path() with empty list raises error."""
    converter = YoloConverter()
    with pytest.raises(EmptyPathListError):
        converter._normalize_input_path([])


def test_yolo_converter_normalize_input_path_with_yaml_path_option(tmp_path: Path) -> None:
    """Test _normalize_input_path() with explicit yaml_path option."""
    converter = YoloConverter()
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("names: [cat, dog]")

    path, _options = converter._normalize_input_path(tmp_path, yaml_path=yaml_file)
    assert path == yaml_file


def test_yolo_converter_normalize_input_path_data_yaml_in_dir(tmp_path: Path) -> None:
    """Test _normalize_input_path() finds data.yaml in directory."""
    converter = YoloConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    yaml_file = dataset_dir / "data.yaml"
    yaml_file.write_text("names: [cat, dog]")

    path, _options = converter._normalize_input_path(dataset_dir)
    assert path == yaml_file


def test_yolo_converter_read_from_yaml_valid(tmp_path: Path) -> None:
    """Test _read_from_yaml() with valid YAML."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    yaml_data = {"path": str(dataset_dir), "names": ["cat", "dog"]}
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    root, data = converter._read_from_yaml(yaml_file)
    assert root == dataset_dir.resolve()
    assert data["names"] == ["cat", "dog"]


def test_yolo_converter_read_from_yaml_relative_path(tmp_path: Path) -> None:
    """Test _read_from_yaml() with relative path."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    yaml_data = {"path": "dataset", "names": ["cat", "dog"]}
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    root, _data = converter._read_from_yaml(yaml_file)
    assert root.exists()
    assert "dataset" in str(root)


def test_yolo_converter_read_from_yaml_no_path(tmp_path: Path) -> None:
    """Test _read_from_yaml() with no path in YAML."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"

    yaml_data = {"names": ["cat", "dog"]}
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    root, _data = converter._read_from_yaml(yaml_file)
    assert root == tmp_path  # Should use YAML file's parent


def test_yolo_converter_read_from_yaml_nonexistent_root(tmp_path: Path) -> None:
    """Test _read_from_yaml() with nonexistent root path."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"

    yaml_data = {"path": "/nonexistent/path", "names": ["cat", "dog"]}
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    with pytest.raises(DatasetRootNotFoundError):
        converter._read_from_yaml(yaml_file)


def test_yolo_converter_load_class_names_from_yaml_list() -> None:
    """Test _load_class_names_from_yaml() with list."""
    converter = YoloConverter()
    data = {"names": ["cat", "dog", "bird"]}
    names = converter._load_class_names_from_yaml(data)
    assert names == ["cat", "dog", "bird"]


def test_yolo_converter_load_class_names_from_yaml_dict() -> None:
    """Test _load_class_names_from_yaml() with dict."""
    converter = YoloConverter()
    data = {"names": {0: "cat", 1: "dog", 2: "bird"}}
    names = converter._load_class_names_from_yaml(data)
    assert names == ["cat", "dog", "bird"]


def test_yolo_converter_load_class_names_from_yaml_dict_string_keys() -> None:
    """Test _load_class_names_from_yaml() with dict with string keys."""
    converter = YoloConverter()
    data = {"names": {"0": "cat", "1": "dog", "2": "bird"}}
    names = converter._load_class_names_from_yaml(data)
    assert names == ["cat", "dog", "bird"]


def test_yolo_converter_load_class_names_from_yaml_missing() -> None:
    """Test _load_class_names_from_yaml() with missing names."""
    converter = YoloConverter()
    data: dict[str, str] = {}
    with pytest.raises(MissingYoloNamesError):
        converter._load_class_names_from_yaml(data)


def test_yolo_converter_load_class_names_from_yaml_unsupported() -> None:
    """Test _load_class_names_from_yaml() with unsupported structure."""
    converter = YoloConverter()
    data = {"names": "invalid"}
    with pytest.raises(UnsupportedYoloNamesStructureError):
        converter._load_class_names_from_yaml(data)


def test_yolo_converter_detect_splits_from_yaml() -> None:
    """Test _detect_splits_from_yaml() method."""
    converter = YoloConverter()
    data = {"train": "images/train", "val": "images/val"}
    splits = converter._detect_splits_from_yaml(data)
    assert splits == ["train", "val"]


def test_yolo_converter_detect_splits_from_yaml_no_splits() -> None:
    """Test _detect_splits_from_yaml() with no splits."""
    converter = YoloConverter()
    data: dict[str, str] = {}
    splits = converter._detect_splits_from_yaml(data)
    assert splits == ["data"]


def test_yolo_converter_autodetect_directory(tmp_path: Path) -> None:
    """Test autodetect() with directory."""
    converter = YoloConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    params = converter.autodetect(dataset_dir)
    assert isinstance(params, YoloDetectedParams)
    assert params.input_path == dataset_dir
    assert params.images_dir == "images"
    assert params.labels_dir == "labels"


def test_yolo_converter_autodetect_yaml_file(tmp_path: Path) -> None:
    """Test autodetect() with YAML file."""
    converter = YoloConverter()
    yaml_file = tmp_path / "data.yaml"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    yaml_data = {"path": str(dataset_dir), "names": ["cat", "dog"]}
    yaml_file.write_text(yaml.safe_dump(yaml_data))

    params = converter.autodetect(yaml_file)
    assert isinstance(params, YoloDetectedParams)
    assert params.yaml_path == yaml_file
    assert params.class_names == ["cat", "dog"]


def test_yolo_converter_autodetect_empty_list() -> None:
    """Test autodetect() with empty list raises error."""
    converter = YoloConverter()
    with pytest.raises(EmptyPathListError):
        converter.autodetect([])


def test_yolo_converter_read_missing_class_names(tmp_path: Path) -> None:
    """Test read() raises error when class names are missing."""
    converter = YoloConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    params = YoloDetectedParams(
        input_path=dataset_dir,
        class_names=None,
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    with pytest.raises(MissingYoloNamesError):
        converter.read(params)


def test_yolo_converter_read_unknown_class_id(tmp_path: Path) -> None:
    """Test read() raises error for unknown class ID."""
    converter = YoloConverter()
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create a test image
    img_path = images_dir / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    # Create label with invalid class ID
    label_path = labels_dir / "test.txt"
    label_path.write_text("5 0.5 0.5 0.1 0.1")  # Class ID 5, but only 2 classes

    params = YoloDetectedParams(
        input_path=dataset_dir,
        class_names=["cat", "dog"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    with pytest.raises(UnknownClassIdError) as exc_info:
        converter.read(params)
    assert exc_info.value.class_id == 5


def test_yolo_converter_write_data_yaml(tmp_path: Path) -> None:
    """Test _write_data_yaml() method."""
    converter = YoloConverter()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    params = YoloWriteDetectedParams(
        class_names=["cat", "dog"],
        splits=["train", "val"],
        images_dir="images",
        labels_dir="labels",
    )

    converter._write_data_yaml(output_dir, params)

    yaml_file = output_dir / "data.yaml"
    assert yaml_file.exists()
    data = yaml.safe_load(yaml_file.read_text())
    assert data["names"] == ["cat", "dog"]
    assert data["nc"] == 2
    assert "train" in data
    assert "val" in data


def test_yolo_converter_write_data_yaml_data_split(tmp_path: Path) -> None:
    """Test _write_data_yaml() with 'data' split."""
    converter = YoloConverter()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    params = YoloWriteDetectedParams(
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )

    converter._write_data_yaml(output_dir, params)

    yaml_file = output_dir / "data.yaml"
    data = yaml.safe_load(yaml_file.read_text())
    assert data["data"] == "images"
