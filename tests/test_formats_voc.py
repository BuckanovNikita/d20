"""Tests for VOC format converter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement

import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from d20.formats.voc import (
    EmptyPathListError,
    InvalidVocBboxError,
    InvalidVocNumericValueError,
    UnknownClassNameError,
    VocConverter,
    _read_split_ids,
)
from d20.types import (
    Annotation,
    Dataset,
    ImageInfo,
    Split,
    VocDetectedParams,
    VocWriteDetectedParams,
)


def test_unknown_class_name_error() -> None:
    """Test UnknownClassNameError exception."""
    error = UnknownClassNameError("unknown_class")
    assert error.name == "unknown_class"
    assert str(error) == "Unknown class name: unknown_class"


def test_empty_path_list_error() -> None:
    """Test EmptyPathListError exception."""
    error = EmptyPathListError()
    assert str(error) == "Empty path list provided for VOC format"


def test_voc_converter_format_name() -> None:
    """Test VocConverter.format_name property."""
    converter = VocConverter()
    assert converter.format_name == "voc"


def test_read_split_ids_valid(tmp_path: Path) -> None:
    """Test _read_split_ids() with valid split file."""
    image_sets_dir = tmp_path / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    split_file = image_sets_dir / "train.txt"
    split_file.write_text("000001\n000002\n000003\n")

    ids = _read_split_ids(image_sets_dir, "train")
    assert ids == ["000001", "000002", "000003"]


def test_read_split_ids_empty_lines(tmp_path: Path) -> None:
    """Test _read_split_ids() handles empty lines."""
    image_sets_dir = tmp_path / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    split_file = image_sets_dir / "train.txt"
    split_file.write_text("000001\n\n000002\n  \n000003\n")

    ids = _read_split_ids(image_sets_dir, "train")
    assert ids == ["000001", "000002", "000003"]


def test_read_split_ids_missing_file(tmp_path: Path) -> None:
    """Test _read_split_ids() with missing file."""
    image_sets_dir = tmp_path / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    ids = _read_split_ids(image_sets_dir, "nonexistent")
    assert ids == []


def test_voc_converter_detect_splits(tmp_path: Path) -> None:
    """Test _detect_splits() method."""
    converter = VocConverter()
    image_sets_dir = tmp_path / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    (image_sets_dir / "train.txt").write_text("000001\n")
    (image_sets_dir / "val.txt").write_text("000002\n")

    splits = converter._detect_splits(image_sets_dir)
    assert "train" in splits
    assert "val" in splits


def test_voc_converter_detect_splits_empty(tmp_path: Path) -> None:
    """Test _detect_splits() with no split files."""
    converter = VocConverter()
    image_sets_dir = tmp_path / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    splits = converter._detect_splits(image_sets_dir)
    assert splits == ["data"]


def test_voc_converter_normalize_input_path_directory(tmp_path: Path) -> None:
    """Test _normalize_input_path() with directory."""
    converter = VocConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    path = converter._normalize_input_path(dataset_dir)
    assert path == dataset_dir


def test_voc_converter_normalize_input_path_list(tmp_path: Path) -> None:
    """Test _normalize_input_path() with list."""
    converter = VocConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    path = converter._normalize_input_path([dataset_dir])
    assert path == dataset_dir


def test_voc_converter_normalize_input_path_empty_list() -> None:
    """Test _normalize_input_path() with empty list raises error."""
    converter = VocConverter()
    with pytest.raises(EmptyPathListError):
        converter._normalize_input_path([])


def test_voc_converter_autodetect_with_splits(tmp_path: Path) -> None:
    """Test autodetect() with ImageSets/Main directory."""
    converter = VocConverter()
    dataset_dir = tmp_path / "dataset"
    image_sets_dir = dataset_dir / "ImageSets" / "Main"
    image_sets_dir.mkdir(parents=True)

    (image_sets_dir / "train.txt").write_text("000001\n")
    (image_sets_dir / "val.txt").write_text("000002\n")

    params = converter.autodetect(dataset_dir)
    assert isinstance(params, VocDetectedParams)
    # Order may vary due to filesystem glob, so check set membership
    assert set(params.splits or []) == {"train", "val"}
    assert params.auto_detect_splits is True
    assert params.images_dir == "JPEGImages"
    assert params.annotations_dir == "Annotations"


def test_voc_converter_autodetect_without_splits(tmp_path: Path) -> None:
    """Test autodetect() without ImageSets/Main directory."""
    converter = VocConverter()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    params = converter.autodetect(dataset_dir)
    assert isinstance(params, VocDetectedParams)
    assert params.splits is None
    assert params.auto_detect_splits is False


def test_voc_converter_autodetect_empty_list() -> None:
    """Test autodetect() with empty list raises error."""
    converter = VocConverter()
    with pytest.raises(EmptyPathListError):
        converter.autodetect([])


def test_voc_converter_parse_image_xml_valid(tmp_path: Path) -> None:
    """Test _parse_image_xml() with valid XML."""
    converter = VocConverter()
    images_dir = tmp_path / "JPEGImages"
    images_dir.mkdir(parents=True)

    # Create a test image
    img_path = images_dir / "000001.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    # Create XML annotation
    xml_path = tmp_path / "000001.xml"
    root = Element("annotation")
    SubElement(root, "filename").text = "000001.jpg"
    size = SubElement(root, "size")
    SubElement(size, "width").text = "100"
    SubElement(size, "height").text = "200"
    SubElement(size, "depth").text = "3"

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    image_info = converter._parse_image_xml(xml_path, "000001", images_dir)
    assert image_info is not None
    assert image_info.image_id == "000001"
    assert image_info.file_name == "000001.jpg"
    assert image_info.width == 100
    assert image_info.height == 200


def test_voc_converter_parse_image_xml_missing_filename(tmp_path: Path) -> None:
    """Test _parse_image_xml() with missing filename."""
    converter = VocConverter()
    images_dir = tmp_path / "JPEGImages"
    images_dir.mkdir(parents=True)

    xml_path = tmp_path / "000001.xml"
    root = Element("annotation")
    # No filename element

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    image_info = converter._parse_image_xml(xml_path, "000001", images_dir)
    assert image_info is None


def test_voc_converter_parse_image_xml_missing_size(tmp_path: Path) -> None:
    """Test _parse_image_xml() with missing size, reads from image."""
    converter = VocConverter()
    images_dir = tmp_path / "JPEGImages"
    images_dir.mkdir(parents=True)

    # Create a test image
    img_path = images_dir / "000001.jpg"
    img = Image.new("RGB", (150, 250), color="blue")
    img.save(img_path)

    xml_path = tmp_path / "000001.xml"
    root = Element("annotation")
    SubElement(root, "filename").text = "000001.jpg"
    # No size element

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    image_info = converter._parse_image_xml(xml_path, "000001", images_dir)
    assert image_info is not None
    assert image_info.width == 150
    assert image_info.height == 250


def test_voc_converter_parse_annotations_from_xml_valid() -> None:
    """Test _parse_annotations_from_xml() with valid XML."""
    converter = VocConverter()
    root = Element("annotation")
    obj1 = SubElement(root, "object")
    SubElement(obj1, "name").text = "cat"
    bbox1 = SubElement(obj1, "bndbox")
    SubElement(bbox1, "xmin").text = "10"
    SubElement(bbox1, "ymin").text = "20"
    SubElement(bbox1, "xmax").text = "50"
    SubElement(bbox1, "ymax").text = "60"

    obj2 = SubElement(root, "object")
    SubElement(obj2, "name").text = "dog"
    bbox2 = SubElement(obj2, "bndbox")
    SubElement(bbox2, "xmin").text = "100"
    SubElement(bbox2, "ymin").text = "110"
    SubElement(bbox2, "xmax").text = "150"
    SubElement(bbox2, "ymax").text = "160"

    annotations = converter._parse_annotations_from_xml(root, "000001", ["cat", "dog"])
    assert len(annotations) == 2
    assert annotations[0].category_id == 0  # cat
    assert annotations[0].bbox == (9, 19, 41, 41)  # xmin-1, ymin-1, xmax-xmin, ymax-ymin
    assert annotations[1].category_id == 1  # dog


def test_voc_converter_parse_annotations_from_xml_unknown_class() -> None:
    """Test _parse_annotations_from_xml() raises error for unknown class."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "unknown_class"
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "10"
    SubElement(bbox, "ymin").text = "20"
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    with pytest.raises(UnknownClassNameError) as exc_info:
        converter._parse_annotations_from_xml(root, "000001", ["cat", "dog"])
    assert exc_info.value.name == "unknown_class"


def test_voc_converter_parse_annotations_from_xml_missing_name() -> None:
    """Test _parse_annotations_from_xml() skips objects without name."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    # No name element
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "10"
    SubElement(bbox, "ymin").text = "20"
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    annotations = converter._parse_annotations_from_xml(root, "000001", ["cat"])
    assert len(annotations) == 0


def test_voc_converter_parse_annotations_from_xml_missing_bbox() -> None:
    """Test _parse_annotations_from_xml() skips objects without bbox."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "cat"
    # No bndbox element

    annotations = converter._parse_annotations_from_xml(root, "000001", ["cat"])
    assert len(annotations) == 0


def test_voc_converter_read_missing_class_names(tmp_path: Path) -> None:
    """Test read() raises error when class names are missing."""
    converter = VocConverter()
    params = VocDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=["train"],
        images_dir="JPEGImages",
        labels_dir=None,
        annotations_dir="Annotations",
        auto_detect_splits=False,
    )

    with pytest.raises(ValueError, match="Class names are required"):
        converter.read(params)


def test_voc_converter_read_empty_list() -> None:
    """Test read() raises error with empty path list."""
    converter = VocConverter()
    params = VocDetectedParams(
        input_path=[],
        class_names=["cat"],
        splits=["train"],
        images_dir="JPEGImages",
        labels_dir=None,
        annotations_dir="Annotations",
        auto_detect_splits=False,
    )

    with pytest.raises(EmptyPathListError):
        converter.read(params)


def test_voc_converter_write(tmp_path: Path) -> None:
    """Test write() method."""
    converter = VocConverter()
    output_dir = tmp_path / "output"

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
    dataset = Dataset(splits={"train": split}, class_names=["cat"])

    params = VocWriteDetectedParams(
        class_names=["cat"],
        splits=["train"],
        images_dir="JPEGImages",
        labels_dir="labels",
        annotations_dir="Annotations",
    )

    converter.write(output_dir, dataset, params)

    # Check output structure
    assert (output_dir / "JPEGImages" / "test.jpg").exists()
    assert (output_dir / "Annotations" / "test.xml").exists()
    assert (output_dir / "ImageSets" / "Main" / "train.txt").exists()

    # Check split file content
    split_file = output_dir / "ImageSets" / "Main" / "train.txt"
    assert split_file.read_text().strip() == "test"


def test_voc_converter_write_split_not_found(tmp_path: Path) -> None:
    """Test write() skips split not found in dataset."""
    converter = VocConverter()
    output_dir = tmp_path / "output"

    dataset = Dataset(splits={}, class_names=["cat"])

    params = VocWriteDetectedParams(
        class_names=["cat"],
        splits=["train"],
        images_dir="JPEGImages",
        labels_dir="labels",
        annotations_dir="Annotations",
    )

    # Should not raise, just skip
    converter.write(output_dir, dataset, params)


# Negative tests for validation failures


def test_voc_parse_invalid_numeric_width(tmp_path: Path) -> None:
    """Test _parse_image_xml() raises error for invalid numeric width."""
    converter = VocConverter()
    xml_path = tmp_path / "test.xml"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create XML with invalid width
    root = Element("annotation")
    SubElement(root, "filename").text = "test.jpg"
    size = SubElement(root, "size")
    SubElement(size, "width").text = "abc"  # Invalid
    SubElement(size, "height").text = "100"

    tree = ET.ElementTree(root)
    tree.write(xml_path)

    with pytest.raises(InvalidVocNumericValueError) as exc_info:
        converter._parse_image_xml(xml_path, "test", images_dir)
    assert exc_info.value.field == "width"
    assert exc_info.value.value == "abc"


def test_voc_parse_invalid_numeric_height(tmp_path: Path) -> None:
    """Test _parse_image_xml() raises error for invalid numeric height."""
    converter = VocConverter()
    xml_path = tmp_path / "test.xml"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create XML with invalid height
    root = Element("annotation")
    SubElement(root, "filename").text = "test.jpg"
    size = SubElement(root, "size")
    SubElement(size, "width").text = "100"
    SubElement(size, "height").text = "xyz"  # Invalid

    tree = ET.ElementTree(root)
    tree.write(xml_path)

    with pytest.raises(InvalidVocNumericValueError) as exc_info:
        converter._parse_image_xml(xml_path, "test", images_dir)
    assert exc_info.value.field == "height"
    assert exc_info.value.value == "xyz"


def test_voc_parse_invalid_bbox_numeric_values() -> None:
    """Test _parse_annotations_from_xml() raises error for invalid numeric bbox values."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "cat"
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "abc"  # Invalid
    SubElement(bbox, "ymin").text = "10"
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    with pytest.raises(InvalidVocNumericValueError) as exc_info:
        converter._parse_annotations_from_xml(root, "test", ["cat", "dog"])
    assert exc_info.value.field == "xmin"
    assert exc_info.value.value == "abc"


def test_voc_parse_invalid_bbox_xmin_ge_xmax() -> None:
    """Test _parse_annotations_from_xml() raises error when xmin >= xmax."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "cat"
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "51"  # xmin > xmax (after -1 conversion: 50 >= 50)
    SubElement(bbox, "ymin").text = "10"
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    with pytest.raises(InvalidVocBboxError) as exc_info:
        converter._parse_annotations_from_xml(root, "test", ["cat", "dog"])
    assert exc_info.value.bbox == (50, 9, 50, 60)  # xmin-1, ymin-1, xmax, ymax


def test_voc_parse_invalid_bbox_ymin_ge_ymax() -> None:
    """Test _parse_annotations_from_xml() raises error when ymin >= ymax."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "cat"
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "10"
    SubElement(bbox, "ymin").text = "61"  # ymin > ymax (after -1 conversion: 60 >= 60)
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    with pytest.raises(InvalidVocBboxError) as exc_info:
        converter._parse_annotations_from_xml(root, "test", ["cat", "dog"])
    assert exc_info.value.bbox == (9, 60, 50, 60)  # xmin-1, ymin-1, xmax, ymax


def test_voc_parse_invalid_bbox_negative_coordinates() -> None:
    """Test _parse_annotations_from_xml() raises error for negative coordinates."""
    converter = VocConverter()
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = "cat"
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "-10"  # Negative
    SubElement(bbox, "ymin").text = "10"
    SubElement(bbox, "xmax").text = "50"
    SubElement(bbox, "ymax").text = "60"

    with pytest.raises(InvalidVocBboxError) as exc_info:
        converter._parse_annotations_from_xml(root, "test", ["cat", "dog"])
    assert exc_info.value.bbox == (-11, 9, 50, 60)  # xmin-1, ymin-1, xmax, ymax


def test_voc_parse_invalid_image_size_zero_dimensions(tmp_path: Path) -> None:
    """Test _parse_image_xml() raises error for zero or negative image dimensions."""
    converter = VocConverter()
    xml_path = tmp_path / "test.xml"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create XML with zero width
    root = Element("annotation")
    SubElement(root, "filename").text = "test.jpg"
    size = SubElement(root, "size")
    SubElement(size, "width").text = "0"  # Invalid
    SubElement(size, "height").text = "100"

    tree = ET.ElementTree(root)
    tree.write(xml_path)

    with pytest.raises(InvalidVocBboxError) as exc_info:
        converter._parse_image_xml(xml_path, "test", images_dir)
    assert exc_info.value.image_size == (0, 100)
