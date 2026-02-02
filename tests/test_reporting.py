"""Tests for reporting/FiftyOne export functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from d20.reporting import _create_dataset_from_splits, _get_format_from_params, export_fiftyone
from d20.types import (
    Annotation,
    CocoDetectedParams,
    Dataset,
    ExportOptions,
    ImageInfo,
    Split,
    YoloDetectedParams,
)


def test_get_format_from_params_yolo(tmp_path: Path) -> None:
    """Test _get_format_from_params() with YOLO params."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    format_name = _get_format_from_params(params)
    assert format_name == "yolo"


def test_get_format_from_params_coco(tmp_path: Path) -> None:
    """Test _get_format_from_params() with COCO params."""
    params = CocoDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=None,
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )
    format_name = _get_format_from_params(params)
    assert format_name == "coco"


def test_get_format_from_params_unknown() -> None:
    """Test _get_format_from_params() raises error for unknown params type."""

    # Create a mock params object that doesn't match any known type
    class UnknownParams:
        pass

    params = UnknownParams()
    # Type ignore because we're testing error handling for invalid types
    with pytest.raises(ValueError, match="Unknown DetectedParams type"):
        _get_format_from_params(params)  # type: ignore[arg-type]


@patch("d20.reporting.fo")
def test_create_dataset_from_splits(mock_fo: MagicMock, tmp_path: Path) -> None:
    """Test _create_dataset_from_splits() creates FiftyOne dataset."""
    # Create a test image
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    images = [
        ImageInfo(image_id="test", file_name="test.jpg", width=100, height=200, path=img_path),
    ]
    annotations = [
        Annotation(image_id="test", category_id=0, bbox=(10, 20, 30, 40)),
    ]
    split = Split(name="train", images=images, annotations=annotations)

    # Mock FiftyOne objects
    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.Detection = MagicMock
    mock_fo.Detections = MagicMock
    mock_fo.Sample = MagicMock

    _create_dataset_from_splits("yolo", split, ["cat", "dog"], "test-dataset")

    mock_fo.Dataset.assert_called_once_with("test-dataset", overwrite=True)
    assert mock_dataset.persistent is True
    assert mock_dataset.add_sample.called


@patch("d20.reporting.fo")
def test_create_dataset_from_splits_missing_image(mock_fo: MagicMock) -> None:
    """Test _create_dataset_from_splits() handles missing images."""
    # Image path doesn't exist
    images = [
        ImageInfo(
            image_id="test",
            file_name="test.jpg",
            width=100,
            height=200,
            path=Path("/nonexistent/test.jpg"),
        ),
    ]
    split = Split(name="train", images=images, annotations=[])

    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.Detection = MagicMock
    mock_fo.Detections = MagicMock
    mock_fo.Sample = MagicMock

    _create_dataset_from_splits("yolo", split, ["cat"], "test-dataset")

    # Should not add sample for missing image
    assert mock_dataset.add_sample.call_count == 0


@patch("d20.reporting.fo")
def test_create_dataset_from_splits_no_annotations(mock_fo: MagicMock, tmp_path: Path) -> None:
    """Test _create_dataset_from_splits() with no annotations."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    images = [
        ImageInfo(image_id="test", file_name="test.jpg", width=100, height=200, path=img_path),
    ]
    split = Split(name="train", images=images, annotations=[])

    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.Detection = MagicMock
    mock_fo.Detections.return_value = None
    mock_fo.Sample = MagicMock

    _create_dataset_from_splits("yolo", split, ["cat"], "test-dataset")

    # Should still add sample even without annotations
    assert mock_dataset.add_sample.called


@patch("d20.reporting.get_converter")
@patch("d20.reporting.fo")
def test_export_fiftyone(mock_fo: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test export_fiftyone() function."""
    # Create test dataset
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    images = [
        ImageInfo(image_id="test", file_name="test.jpg", width=100, height=200, path=img_path),
    ]
    annotations = [
        Annotation(image_id="test", category_id=0, bbox=(10, 20, 30, 40)),
    ]
    split = Split(name="train", images=images, annotations=annotations)
    dataset = Dataset(splits={"train": split}, class_names=["cat"])

    # Mock converter
    mock_converter = MagicMock()
    mock_converter.read.return_value = dataset
    mock_get_converter.return_value = mock_converter

    # Mock FiftyOne
    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.list_datasets.return_value = ["test-dataset"]
    mock_fo.Detection = MagicMock
    mock_fo.Detections = MagicMock
    mock_fo.Sample = MagicMock

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    options = ExportOptions(split=None)

    export_fiftyone(params, options)

    mock_converter.read.assert_called_once_with(params)
    mock_fo.Dataset.assert_called()


@patch("d20.reporting.get_converter")
@patch("d20.reporting.fo")
def test_export_fiftyone_specific_split(mock_fo: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test export_fiftyone() with specific split."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    images = [
        ImageInfo(image_id="test", file_name="test.jpg", width=100, height=200, path=img_path),
    ]
    split1 = Split(name="train", images=images, annotations=[])
    split2 = Split(name="val", images=[], annotations=[])
    dataset = Dataset(splits={"train": split1, "val": split2}, class_names=["cat"])

    mock_converter = MagicMock()
    mock_converter.read.return_value = dataset
    mock_get_converter.return_value = mock_converter

    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.list_datasets.return_value = ["test-dataset"]
    mock_fo.Detection = MagicMock
    mock_fo.Detections = MagicMock
    mock_fo.Sample = MagicMock

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train", "val"],
        images_dir="images",
        labels_dir="labels",
    )
    options = ExportOptions(split="train")

    export_fiftyone(params, options)

    # Should only export train split
    assert mock_fo.Dataset.call_count == 1


@patch("d20.reporting.get_converter")
@patch("d20.reporting.fo")
def test_export_fiftyone_split_not_found(mock_fo: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test export_fiftyone() with split not found."""
    dataset = Dataset(splits={}, class_names=["cat"])

    mock_converter = MagicMock()
    mock_converter.read.return_value = dataset
    mock_get_converter.return_value = mock_converter

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    options = ExportOptions(split="nonexistent")

    # Should not raise, just return early
    export_fiftyone(params, options)
    assert not mock_fo.Dataset.called


@patch("d20.reporting.get_converter")
@patch("d20.reporting.fo")
def test_export_fiftyone_no_class_names(mock_fo: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test export_fiftyone() with no class names."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    images = [
        ImageInfo(image_id="test", file_name="test.jpg", width=100, height=200, path=img_path),
    ]
    split = Split(name="train", images=images, annotations=[])
    dataset = Dataset(splits={"train": split}, class_names=[])

    mock_converter = MagicMock()
    mock_converter.read.return_value = dataset
    mock_get_converter.return_value = mock_converter

    mock_dataset = MagicMock()
    mock_fo.Dataset.return_value = mock_dataset
    mock_fo.list_datasets.return_value = ["test-dataset"]
    mock_fo.Detection = MagicMock
    mock_fo.Detections = MagicMock
    mock_fo.Sample = MagicMock

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    options = ExportOptions(split=None)

    # Should not raise, but log warning
    export_fiftyone(params, options)
    assert mock_fo.Dataset.called


@patch("d20.reporting.get_converter")
@patch("d20.reporting.fo")
def test_export_fiftyone_empty_split(mock_fo: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test export_fiftyone() skips empty splits."""
    split = Split(name="train", images=[], annotations=[])
    dataset = Dataset(splits={"train": split}, class_names=["cat"])

    mock_converter = MagicMock()
    mock_converter.read.return_value = dataset
    mock_get_converter.return_value = mock_converter

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    options = ExportOptions(split=None)

    export_fiftyone(params, options)

    # Should not create dataset for empty split
    assert not mock_fo.Dataset.called
