"""Tests for COCO format converter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from d20.formats.coco import (
    COCOCategoryMissingNameError,
    CocoConverter,
    InvalidCocoBboxError,
    InvalidCocoCategoryIdError,
    InvalidCocoImageIdError,
    InvalidCocoJsonError,
    SplitNamesMismatchError,
    UnknownClassNameError,
    _extract_categories_from_json_file,
)
from d20.types import (
    Annotation,
    CocoDetectedParams,
    CocoWriteDetectedParams,
    Dataset,
    ImageInfo,
    Split,
)


def test_coco_category_missing_name_error() -> None:
    """Test COCOCategoryMissingNameError exception."""
    error = COCOCategoryMissingNameError()
    assert str(error) == "COCO category missing name"


def test_unknown_class_name_error() -> None:
    """Test UnknownClassNameError exception."""
    error = UnknownClassNameError("unknown_class")
    assert error.name == "unknown_class"
    assert str(error) == "Unknown class name: unknown_class"


def test_split_names_mismatch_error() -> None:
    """Test SplitNamesMismatchError exception."""
    error = SplitNamesMismatchError()
    assert str(error) == "Number of split names must match number of JSON files"


def test_coco_converter_format_name() -> None:
    """Test CocoConverter.format_name property."""
    converter = CocoConverter()
    assert converter.format_name == "coco"


def test_extract_categories_from_json_file_valid(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() with valid JSON."""
    json_file = tmp_path / "annotations.json"
    json_data = {
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bird"},
        ],
    }
    json_file.write_text(json.dumps(json_data))

    names, mapping = _extract_categories_from_json_file(json_file)
    assert names == {"cat", "dog", "bird"}
    assert mapping == {"cat": 1, "dog": 2, "bird": 3}


def test_extract_categories_from_json_file_empty(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() with empty categories."""
    json_file = tmp_path / "annotations.json"
    json_data: dict[str, list[dict[str, int | str]]] = {"categories": []}
    json_file.write_text(json.dumps(json_data))

    names, mapping = _extract_categories_from_json_file(json_file)
    assert names == set()
    assert mapping == {}


def test_extract_categories_from_json_file_invalid_json(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() raises error for invalid JSON."""
    json_file = tmp_path / "annotations.json"
    json_file.write_text("invalid json content")

    # Should raise InvalidCocoJsonError
    with pytest.raises(InvalidCocoJsonError) as exc_info:
        _extract_categories_from_json_file(json_file)
    assert exc_info.value.json_path == json_file


def test_extract_categories_from_json_file_missing_categories(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() with missing categories key."""
    json_file = tmp_path / "annotations.json"
    json_data: dict[str, list[dict[str, int | str]]] = {"images": []}
    json_file.write_text(json.dumps(json_data))

    names, mapping = _extract_categories_from_json_file(json_file)
    assert names == set()
    assert mapping == {}


def test_coco_converter_autodetect_directory(tmp_path: Path) -> None:
    """Test autodetect() with directory."""
    converter = CocoConverter()
    dataset_dir = tmp_path / "dataset"
    annotations_dir = dataset_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    json_file = annotations_dir / "train.json"
    json_data = {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [],
        "annotations": [],
    }
    json_file.write_text(json.dumps(json_data))

    params = converter.autodetect(dataset_dir)
    assert isinstance(params, CocoDetectedParams)
    assert params.splits == ["train"]
    assert params.class_names == ["cat"]


def test_coco_converter_autodetect_list_of_files(tmp_path: Path) -> None:
    """Test autodetect() with list of JSON files."""
    converter = CocoConverter()
    json1 = tmp_path / "train.json"
    json2 = tmp_path / "val.json"

    json1.write_text(json.dumps({"categories": [{"id": 1, "name": "cat"}]}))
    json2.write_text(json.dumps({"categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]}))

    params = converter.autodetect([json1, json2])
    assert isinstance(params, CocoDetectedParams)
    assert params.splits == ["train", "val"]
    assert params.class_names is not None
    assert "cat" in params.class_names
    assert "dog" in params.class_names


def test_coco_converter_read_split_from_json_missing_file(tmp_path: Path) -> None:
    """Test _read_split_from_json() with missing file."""
    converter = CocoConverter()
    json_file = tmp_path / "nonexistent.json"

    split = converter._read_split_from_json(
        json_file,
        tmp_path,
        ["cat"],
        "images",
        "train",
    )
    assert split.name == "train"
    assert split.images == []
    assert split.annotations == []


def test_coco_converter_read_split_from_json_missing_category_name(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for missing category name."""
    converter = CocoConverter()
    json_file = tmp_path / "annotations.json"
    json_data = {
        "categories": [{"id": 1}],  # Missing name
        "images": [],
        "annotations": [],
    }
    json_file.write_text(json.dumps(json_data))

    with pytest.raises(COCOCategoryMissingNameError):
        converter._read_split_from_json(
            json_file,
            tmp_path,
            ["cat"],
            "images",
            "train",
        )


def test_coco_converter_read_split_from_json_unknown_class_name(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for unknown class name."""
    converter = CocoConverter()
    json_file = tmp_path / "annotations.json"
    json_data = {
        "categories": [{"id": 1, "name": "unknown_class"}],
        "images": [],
        "annotations": [],
    }
    json_file.write_text(json.dumps(json_data))

    with pytest.raises(UnknownClassNameError) as exc_info:
        converter._read_split_from_json(
            json_file,
            tmp_path,
            ["cat", "dog"],
            "images",
            "train",
        )
    assert exc_info.value.name == "unknown_class"


def test_coco_converter_read_split_names_mismatch(tmp_path: Path) -> None:
    """Test read() raises error when split names don't match file count."""
    converter = CocoConverter()
    json_file = tmp_path / "test.json"

    params = CocoDetectedParams(
        input_path=[json_file],
        class_names=["cat"],
        splits=["train", "val"],  # 2 splits but only 1 file
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )

    with pytest.raises(SplitNamesMismatchError):
        converter.read(params)


def test_coco_converter_read_missing_class_names(tmp_path: Path) -> None:
    """Test read() raises error when class names are missing."""
    converter = CocoConverter()
    params = CocoDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=["train"],
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )

    with pytest.raises(ValueError, match="Class names are required"):
        converter.read(params)


def test_coco_converter_write(tmp_path: Path) -> None:
    """Test write() method."""
    converter = CocoConverter()
    output_dir = tmp_path / "output"

    images = [
        ImageInfo(image_id="1", file_name="img1.jpg", width=100, height=200, path=tmp_path / "img1.jpg"),
    ]
    annotations = [
        Annotation(image_id="1", category_id=0, bbox=(10, 20, 30, 40)),
    ]
    split = Split(name="train", images=images, annotations=annotations)
    dataset = Dataset(splits={"train": split}, class_names=["cat", "dog"])

    # Create a dummy image file
    img_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    params = CocoWriteDetectedParams(
        class_names=["cat", "dog"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
        annotations_dir="annotations",
    )

    converter.write(output_dir, dataset, params)

    annotations_file = output_dir / "annotations" / "train.json"
    assert annotations_file.exists()

    data = json.loads(annotations_file.read_text())
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    assert len(data["categories"]) == 2
    assert data["categories"][0]["name"] == "cat"
    assert data["categories"][1]["name"] == "dog"


def test_coco_converter_write_split_not_found(tmp_path: Path) -> None:
    """Test write() skips split not found in dataset."""
    converter = CocoConverter()
    output_dir = tmp_path / "output"

    dataset = Dataset(splits={}, class_names=["cat"])

    params = CocoWriteDetectedParams(
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
        annotations_dir="annotations",
    )

    # Should not raise, just skip
    converter.write(output_dir, dataset, params)


# Negative tests for validation failures


def test_extract_categories_invalid_json_raises_error(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() raises error for invalid JSON."""
    json_path = tmp_path / "invalid.json"
    json_path.write_text("{ invalid json }")

    with pytest.raises(InvalidCocoJsonError) as exc_info:
        _extract_categories_from_json_file(json_path)
    assert exc_info.value.json_path == json_path


def test_extract_categories_missing_file_raises_error(tmp_path: Path) -> None:
    """Test _extract_categories_from_json_file() raises error for missing file."""
    json_path = tmp_path / "nonexistent.json"

    with pytest.raises(InvalidCocoJsonError) as exc_info:
        _extract_categories_from_json_file(json_path)
    assert exc_info.value.json_path == json_path


def test_coco_read_invalid_category_id(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for invalid category_id."""
    converter = CocoConverter()
    json_path = tmp_path / "train.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create JSON with annotation referencing non-existent category_id
    data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 999,  # Non-existent category_id
                "bbox": [10, 10, 20, 20],
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    json_path.write_text(json.dumps(data))

    with pytest.raises(InvalidCocoCategoryIdError) as exc_info:
        converter._read_split_from_json(json_path, tmp_path, ["cat"], "images", "train")
    assert exc_info.value.category_id == 999
    assert exc_info.value.available_ids == [1]


def test_coco_read_invalid_image_id(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for invalid image_id in annotation."""
    converter = CocoConverter()
    json_path = tmp_path / "train.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create JSON with annotation referencing non-existent image_id
    data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 999,  # Non-existent image_id
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    json_path.write_text(json.dumps(data))

    with pytest.raises(InvalidCocoImageIdError) as exc_info:
        converter._read_split_from_json(json_path, tmp_path, ["cat"], "images", "train")
    assert exc_info.value.image_id == "999"


def test_coco_read_invalid_bbox_wrong_length(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for bbox with wrong length."""
    converter = CocoConverter()
    json_path = tmp_path / "train.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create JSON with invalid bbox (wrong length)
    data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20],  # Only 3 elements, should be 4
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    json_path.write_text(json.dumps(data))

    with pytest.raises(InvalidCocoBboxError) as exc_info:
        converter._read_split_from_json(json_path, tmp_path, ["cat"], "images", "train")
    assert exc_info.value.bbox == [10, 10, 20]


def test_coco_read_invalid_bbox_negative_dimensions(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for bbox with negative dimensions."""
    converter = CocoConverter()
    json_path = tmp_path / "train.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create JSON with invalid bbox (negative width)
    data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, -20, 20],  # Negative width
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    json_path.write_text(json.dumps(data))

    with pytest.raises(InvalidCocoBboxError) as exc_info:
        converter._read_split_from_json(json_path, tmp_path, ["cat"], "images", "train")
    assert exc_info.value.bbox == [10, 10, -20, 20]
    assert exc_info.value.image_size == (100, 100)


def test_coco_read_invalid_bbox_out_of_bounds(tmp_path: Path) -> None:
    """Test _read_split_from_json() raises error for bbox outside image bounds."""
    converter = CocoConverter()
    json_path = tmp_path / "train.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create JSON with bbox extending beyond image bounds
    data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [90, 10, 20, 20],  # x + width = 110 > 100
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    json_path.write_text(json.dumps(data))

    with pytest.raises(InvalidCocoBboxError) as exc_info:
        converter._read_split_from_json(json_path, tmp_path, ["cat"], "images", "train")
    assert exc_info.value.bbox == [90, 10, 20, 20]
    assert exc_info.value.image_size == (100, 100)


def test_coco_write_invalid_image_id(tmp_path: Path) -> None:
    """Test write() raises error for annotation with invalid image_id."""
    converter = CocoConverter()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a dummy image file
    img_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    # Create dataset with annotation referencing non-existent image
    images = [
        ImageInfo(
            image_id="1",
            file_name="img1.jpg",
            width=100,
            height=100,
            path=img_path,
        )
    ]
    annotations = [
        Annotation(
            image_id="999",  # Non-existent image_id
            category_id=0,
            bbox=(10, 10, 20, 20),
        )
    ]
    split = Split(name="train", images=images, annotations=annotations)
    dataset = Dataset(splits={"train": split}, class_names=["cat"])

    params = CocoWriteDetectedParams(
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
        annotations_dir="annotations",
    )

    with pytest.raises(InvalidCocoImageIdError) as exc_info:
        converter.write(output_dir, dataset, params)
    assert exc_info.value.image_id == "999"
