"""Tests for dataset format conversions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from defusedxml import ElementTree

from d20.convert import ConversionRequest, convert_dataset
from d20.formats import get_converter
from d20.types import (
    CocoDetectedParams,
    CocoWriteDetectedParams,
    VocDetectedParams,
    VocWriteDetectedParams,
    YoloDetectedParams,
    YoloWriteDetectedParams,
)


@dataclass
class WriteParamsConfig:
    """Configuration for building WriteDetectedParams."""

    class_names: list[str]
    splits: list[str]
    images_dir: str
    labels_dir: str
    annotations_dir: str = "annotations"


IMAGE_EXTS = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "datasets"
DATASET_YAMLS = (
    FIXTURES_DIR / "coco8.yaml",
    FIXTURES_DIR / "coco12-formats.yaml",
)


@dataclass(frozen=True)
class YoloFixture:
    """Fixture data for YOLO dataset tests."""

    name: str
    root: Path
    yaml_path: Path
    data: dict[str, Any]


def _resolve_yolo_fixture(yaml_path: Path) -> YoloFixture:
    assert yaml_path.exists(), f"Missing required fixture yaml: {yaml_path}"

    data = yaml.safe_load(yaml_path.read_text()) or {}

    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent

    assert root.exists(), f"Missing required fixture path: {root}"

    return YoloFixture(name=root.name, root=root, yaml_path=yaml_path, data=data)


def _load_class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    assert names, "Missing 'names' in YOLO YAML"

    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        assert all(isinstance(k, (str, int)) for k in names), "Invalid key types in names dict"

        def _key_as_int(key: str | int) -> int:
            return int(key)

        return [names[key] for key in sorted(names.keys(), key=_key_as_int)]

    pytest.fail("Unsupported 'names' structure in YOLO YAML")


def _detect_splits(fixture_root: Path, images_dir: str) -> list[str]:
    base_dir = fixture_root / images_dir
    splits = [split for split in ("train", "val", "test") if (base_dir / split).exists()]
    return splits or ["data"]


def _split_path(root: Path, base_dir: str, split: str) -> Path:
    split_dir = root / base_dir / split
    return split_dir if split_dir.exists() else root / base_dir


def _count_images(images_dir: Path) -> int:
    return sum(1 for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _count_yolo_annotations(labels_dir: Path) -> int:
    total = 0
    for path in labels_dir.rglob("*.txt"):
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        total += len(lines)
    return total


def _count_coco_annotations(labels_path: Path) -> int:
    data = json.loads(labels_path.read_text())
    return len(data.get("annotations", []))


def _read_coco_image_count(labels_path: Path) -> int:
    data = json.loads(labels_path.read_text())
    return len(data.get("images", []))


def _read_coco_category_count(labels_path: Path) -> int:
    data = json.loads(labels_path.read_text())
    return len(data.get("categories", []))


def _read_voc_split_ids(image_sets_dir: Path, split: str) -> list[str]:
    split_path = image_sets_dir / f"{split}.txt"
    assert split_path.exists(), f"Missing VOC split file: {split_path}"

    ids = []
    for line in split_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        ids.append(stripped.split()[0])
    return ids


def _count_voc_annotations(annotations_dir: Path, split_ids: list[str]) -> int:
    total = 0
    for image_id in split_ids:
        xml_path = annotations_dir / f"{image_id}.xml"
        tree = ElementTree.parse(xml_path)
        total += len(tree.findall(".//object"))
    return total


def _build_write_params(
    format_name: str,
    config: WriteParamsConfig,
) -> CocoWriteDetectedParams | YoloWriteDetectedParams | VocWriteDetectedParams:
    """Build WriteDetectedParams for testing."""
    if format_name == "yolo":
        return YoloWriteDetectedParams(
            class_names=config.class_names,
            splits=config.splits,
            images_dir=config.images_dir,
            labels_dir=config.labels_dir,
            annotations_dir=config.annotations_dir,
        )
    if format_name == "coco":
        return CocoWriteDetectedParams(
            class_names=config.class_names,
            splits=config.splits,
            images_dir=config.images_dir,
            labels_dir=config.labels_dir,
            annotations_dir=config.annotations_dir,
        )
    if format_name == "voc":
        return VocWriteDetectedParams(
            class_names=config.class_names,
            splits=config.splits,
            images_dir=config.images_dir,
            labels_dir=config.labels_dir,
            annotations_dir=config.annotations_dir,
        )
    msg = f"Unknown format: {format_name}"
    raise ValueError(msg)


@dataclass
class VerificationConfig:
    """Configuration for verification."""

    fixture_root: Path
    splits: list[str]
    images_dir: str
    labels_dir: str
    class_names: list[str] | None = None


def _verify_coco_conversion(coco_dir: Path, config: VerificationConfig) -> None:
    """Verify COCO conversion results."""
    for split in config.splits:
        input_images_dir = _split_path(config.fixture_root, config.images_dir, split)
        input_labels_dir = _split_path(config.fixture_root, config.labels_dir, split)
        expected_images = _count_images(input_images_dir)
        expected_annotations = _count_yolo_annotations(input_labels_dir)

        coco_images_dir = _split_path(coco_dir, config.images_dir, split)
        coco_labels_path = coco_dir / "annotations" / f"{split}.json"

        assert coco_images_dir.exists()
        assert coco_labels_path.exists()
        assert _read_coco_image_count(coco_labels_path) == expected_images
        assert _count_coco_annotations(coco_labels_path) == expected_annotations
        if config.class_names:
            assert _read_coco_category_count(coco_labels_path) == len(config.class_names)


def _verify_voc_conversion(voc_dir: Path, config: VerificationConfig) -> None:
    """Verify VOC conversion results."""
    voc_images_dir = voc_dir / "JPEGImages"
    voc_annotations_dir = voc_dir / "Annotations"
    voc_image_sets_dir = voc_dir / "ImageSets" / "Main"

    assert voc_images_dir.exists()
    assert voc_annotations_dir.exists()
    assert voc_image_sets_dir.exists()

    for split in config.splits:
        split_ids = _read_voc_split_ids(voc_image_sets_dir, split)
        expected_images = _count_images(_split_path(config.fixture_root, config.images_dir, split))
        expected_annotations = _count_yolo_annotations(
            _split_path(config.fixture_root, config.labels_dir, split),
        )

        assert len(split_ids) == expected_images
        assert _count_voc_annotations(voc_annotations_dir, split_ids) == expected_annotations


def _verify_yolo_conversion(yolo_dir: Path, config: VerificationConfig) -> None:
    """Verify YOLO conversion results."""
    for split in config.splits:
        input_images_dir = _split_path(config.fixture_root, config.images_dir, split)
        input_labels_dir = _split_path(config.fixture_root, config.labels_dir, split)
        expected_images = _count_images(input_images_dir)
        expected_annotations = _count_yolo_annotations(input_labels_dir)

        yolo_images_dir = _split_path(yolo_dir, config.images_dir, split)
        yolo_labels_dir = _split_path(yolo_dir, config.labels_dir, split)

        assert yolo_images_dir.exists()
        assert yolo_labels_dir.exists()
        assert _count_images(yolo_images_dir) == expected_images
        assert _count_yolo_annotations(yolo_labels_dir) == expected_annotations


@pytest.mark.parametrize("yaml_path", DATASET_YAMLS)
def test_yolo_coco_voc_roundtrip(tmp_path: Path, yaml_path: Path) -> None:
    """Test roundtrip conversion: YOLO -> COCO -> VOC -> YOLO."""
    fixture = _resolve_yolo_fixture(yaml_path)
    fixture_root = fixture.root
    data = fixture.data

    class_names = _load_class_names(data)
    images_dir = "images"
    labels_dir = "labels"
    splits = _detect_splits(fixture_root, images_dir)

    # Autodetect YOLO parameters
    yolo_converter = get_converter("yolo")
    detected = yolo_converter.autodetect(fixture_root)
    # Type narrowing: autodetect returns YoloDetectedParams for yolo converter
    if not isinstance(detected, YoloDetectedParams):
        msg = f"Expected YoloDetectedParams, got {type(detected)}"
        raise TypeError(msg)
    read_params = YoloDetectedParams(
        input_path=detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
        annotations_dir=detected.annotations_dir,
        yaml_path=detected.yaml_path,
        dataset_root=detected.dataset_root,
    )

    config = WriteParamsConfig(class_names=class_names, splits=splits, images_dir=images_dir, labels_dir=labels_dir)
    write_params = _build_write_params("coco", config)

    coco_dir = tmp_path / "coco"
    request = ConversionRequest(
        input_format="yolo",
        output_format="coco",
        input_path=fixture_root,
        output_dir=coco_dir,
        read_params=read_params,
        write_params=write_params,
    )
    convert_dataset(request)
    coco_config = VerificationConfig(
        fixture_root=fixture_root,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
    )
    _verify_coco_conversion(coco_dir, coco_config)

    # Autodetect COCO parameters
    coco_converter = get_converter("coco")
    coco_detected = coco_converter.autodetect(coco_dir)
    # Type narrowing: autodetect returns CocoDetectedParams for coco converter
    if not isinstance(coco_detected, CocoDetectedParams):
        msg = f"Expected CocoDetectedParams, got {type(coco_detected)}"
        raise TypeError(msg)
    coco_read_params = CocoDetectedParams(
        input_path=coco_detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=coco_detected.labels_dir,
        annotations_dir="annotations",
        split_files=coco_detected.split_files,
    )

    voc_config = WriteParamsConfig(class_names=class_names, splits=splits, images_dir=images_dir, labels_dir=labels_dir)
    voc_write_params = _build_write_params("voc", voc_config)

    voc_dir = tmp_path / "voc"
    request = ConversionRequest(
        input_format="coco",
        output_format="voc",
        input_path=coco_dir,
        output_dir=voc_dir,
        read_params=coco_read_params,
        write_params=voc_write_params,
    )
    convert_dataset(request)
    voc_config_verify = VerificationConfig(
        fixture_root=fixture_root,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
    )
    _verify_voc_conversion(voc_dir, voc_config_verify)

    # Autodetect VOC parameters
    voc_converter = get_converter("voc")
    voc_detected = voc_converter.autodetect(voc_dir)
    # Type narrowing: autodetect returns VocDetectedParams for voc converter
    if not isinstance(voc_detected, VocDetectedParams):
        msg = f"Expected VocDetectedParams, got {type(voc_detected)}"
        raise TypeError(msg)
    voc_read_params = VocDetectedParams(
        input_path=voc_detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=voc_detected.images_dir,
        labels_dir=voc_detected.labels_dir,
        annotations_dir=voc_detected.annotations_dir,
        auto_detect_splits=voc_detected.auto_detect_splits,
    )

    yolo_config = WriteParamsConfig(
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
    )
    yolo_write_params = _build_write_params("yolo", yolo_config)

    yolo_dir = tmp_path / "yolo"
    request = ConversionRequest(
        input_format="voc",
        output_format="yolo",
        input_path=voc_dir,
        output_dir=yolo_dir,
        read_params=voc_read_params,
        write_params=yolo_write_params,
    )
    convert_dataset(request)
    yolo_config_verify = VerificationConfig(
        fixture_root=fixture_root,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
    )
    _verify_yolo_conversion(yolo_dir, yolo_config_verify)


@pytest.mark.parametrize("yaml_path", DATASET_YAMLS)
def test_yolo_read_from_yaml(tmp_path: Path, yaml_path: Path) -> None:
    """Test reading YOLO dataset directly from YAML file."""
    fixture = _resolve_yolo_fixture(yaml_path)
    data = fixture.data

    class_names = _load_class_names(data)
    images_dir = "images"
    labels_dir = "labels"
    splits = _detect_splits(fixture.root, images_dir)

    # Autodetect YOLO parameters from YAML file
    yolo_converter = get_converter("yolo")
    detected = yolo_converter.autodetect(yaml_path)
    # Type narrowing: autodetect returns YoloDetectedParams for yolo converter
    if not isinstance(detected, YoloDetectedParams):
        msg = f"Expected YoloDetectedParams, got {type(detected)}"
        raise TypeError(msg)
    read_params = YoloDetectedParams(
        input_path=detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
        annotations_dir=detected.annotations_dir,
        yaml_path=detected.yaml_path,
        dataset_root=detected.dataset_root,
    )

    config = WriteParamsConfig(class_names=class_names, splits=splits, images_dir=images_dir, labels_dir=labels_dir)
    write_params = _build_write_params("coco", config)

    # Test reading from YAML file
    coco_dir = tmp_path / "coco_from_yaml"
    request = ConversionRequest(
        input_format="yolo",
        output_format="coco",
        input_path=yaml_path,
        output_dir=coco_dir,
        read_params=read_params,
        write_params=write_params,
    )
    convert_dataset(request)

    # Verify conversion worked
    for split in splits:
        coco_labels_path = coco_dir / "annotations" / f"{split}.json"
        assert coco_labels_path.exists()
        assert _read_coco_image_count(coco_labels_path) > 0


@pytest.mark.parametrize("yaml_path", DATASET_YAMLS)
def test_coco_yolo_direct_conversion(tmp_path: Path, yaml_path: Path) -> None:
    """Test direct COCO to YOLO conversion."""
    fixture = _resolve_yolo_fixture(yaml_path)
    fixture_root = fixture.root
    data = fixture.data

    class_names = _load_class_names(data)
    images_dir = "images"
    labels_dir = "labels"
    splits = _detect_splits(fixture_root, images_dir)

    # Convert YOLO to COCO first
    yolo_converter = get_converter("yolo")
    detected = yolo_converter.autodetect(fixture_root)
    if not isinstance(detected, YoloDetectedParams):
        msg = f"Expected YoloDetectedParams, got {type(detected)}"
        raise TypeError(msg)
    read_params = YoloDetectedParams(
        input_path=detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
        annotations_dir=detected.annotations_dir,
        yaml_path=detected.yaml_path,
        dataset_root=detected.dataset_root,
    )

    config = WriteParamsConfig(class_names=class_names, splits=splits, images_dir=images_dir, labels_dir=labels_dir)
    write_params = _build_write_params("coco", config)

    coco_dir = tmp_path / "coco"
    request = ConversionRequest(
        input_format="yolo",
        output_format="coco",
        input_path=fixture_root,
        output_dir=coco_dir,
        read_params=read_params,
        write_params=write_params,
    )
    convert_dataset(request)

    # Now convert COCO to YOLO
    coco_converter = get_converter("coco")
    coco_detected = coco_converter.autodetect(coco_dir)
    if not isinstance(coco_detected, CocoDetectedParams):
        msg = f"Expected CocoDetectedParams, got {type(coco_detected)}"
        raise TypeError(msg)
    coco_read_params = CocoDetectedParams(
        input_path=coco_detected.input_path,
        class_names=class_names,
        splits=splits,
        images_dir=images_dir,
        labels_dir=coco_detected.labels_dir,
        annotations_dir="annotations",
        split_files=coco_detected.split_files,
    )

    yolo_write_params = _build_write_params("yolo", config)

    yolo_dir = tmp_path / "yolo_from_coco"
    request = ConversionRequest(
        input_format="coco",
        output_format="yolo",
        input_path=coco_dir,
        output_dir=yolo_dir,
        read_params=coco_read_params,
        write_params=yolo_write_params,
    )
    convert_dataset(request)

    # Verify YOLO conversion
    yolo_config_verify = VerificationConfig(
        fixture_root=fixture_root,
        splits=splits,
        images_dir=images_dir,
        labels_dir=labels_dir,
    )
    _verify_yolo_conversion(yolo_dir, yolo_config_verify)
