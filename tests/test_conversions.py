from __future__ import annotations

import json
from pathlib import Path

from defusedxml import ElementTree as ET
import pytest
import yaml

from d20.config import load_config
from d20.convert import convert_dataset
from d20.reporting import generate_fiftyone_report

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
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DATASET_NAMES = ("coco8", "coco12-formats")


def _require_fixture_dir(name: str) -> Path:
    path = FIXTURES_DIR / name
    if not path.exists():
        raise AssertionError(f"Missing required fixture path: {path}")
    return path


def _find_yaml(fixture_root: Path) -> Path:
    preferred = fixture_root / "data.yaml"
    if preferred.exists():
        return preferred

    sibling = fixture_root.with_suffix(".yaml")
    if sibling.exists():
        return sibling

    candidates = sorted(fixture_root.parent.glob("*.yaml"))
    if not candidates:
        raise AssertionError(f"No YAML config found near {fixture_root}")
    return candidates[0]


def _load_class_names(data: dict) -> list[str]:
    names = data.get("names")
    if not names:
        raise AssertionError("Missing 'names' in YOLO YAML")

    if isinstance(names, list):
        return names

    if isinstance(names, dict):

        def _key_as_int(key: object) -> int:
            return int(key)

        return [names[key] for key in sorted(names.keys(), key=_key_as_int)]

    raise AssertionError("Unsupported 'names' structure in YOLO YAML")


def _detect_splits(fixture_root: Path, images_dir: str) -> list[str]:
    base_dir = fixture_root / images_dir
    splits = [split for split in ("train", "val", "test") if (base_dir / split).exists()]
    return splits or ["data"]


def _split_path(root: Path, base_dir: str, split: str) -> Path:
    split_dir = root / base_dir / split
    return split_dir if split_dir.exists() else root / base_dir


def _count_images(images_dir: Path) -> int:
    return sum(
        1 for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


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
    if not split_path.exists():
        raise AssertionError(f"Missing VOC split file: {split_path}")

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
        tree = ET.parse(xml_path)
        total += len(tree.findall(".//object"))
    return total


def _write_config(
    tmp_path: Path,
    class_names: list[str],
    splits: list[str],
    images_dir: str,
    labels_dir: str,
) -> Path:
    config_path = tmp_path / "conversion.yaml"
    payload = {
        "class_names": class_names,
        "splits": splits,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "annotations_dir": "annotations",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_yolo_coco_voc_roundtrip(tmp_path: Path, dataset_name: str) -> None:
    fixture_root = _require_fixture_dir(dataset_name)
    data_yaml = _find_yaml(fixture_root)
    data = yaml.safe_load(data_yaml.read_text())

    class_names = _load_class_names(data)
    images_dir = "images"
    labels_dir = "labels"
    splits = _detect_splits(fixture_root, images_dir)
    config_path = _write_config(tmp_path, class_names, splits, images_dir, labels_dir)
    config = load_config(config_path)

    coco_dir = tmp_path / "coco"
    convert_dataset("yolo", "coco", fixture_root, coco_dir, config)

    for split in config.splits:
        input_images_dir = _split_path(fixture_root, config.images_dir, split)
        input_labels_dir = _split_path(fixture_root, config.labels_dir, split)
        expected_images = _count_images(input_images_dir)
        expected_annotations = _count_yolo_annotations(input_labels_dir)

        coco_images_dir = _split_path(coco_dir, config.images_dir, split)
        coco_labels_path = coco_dir / config.annotations_dir / f"{split}.json"

        assert coco_images_dir.exists()
        assert coco_labels_path.exists()
        assert _read_coco_image_count(coco_labels_path) == expected_images
        assert _count_coco_annotations(coco_labels_path) == expected_annotations
        assert _read_coco_category_count(coco_labels_path) == len(config.class_names)

    coco_report_dir = REPORTS_DIR / dataset_name / "yolo_to_coco"
    coco_report_path = generate_fiftyone_report(
        dataset_format="coco",
        output_dir=coco_dir,
        config=config,
        report_dir=coco_report_dir,
        split=config.splits[0],
    )
    assert coco_report_path.exists()

    voc_dir = tmp_path / "voc"
    convert_dataset("coco", "voc", coco_dir, voc_dir, config)

    voc_images_dir = voc_dir / "JPEGImages"
    voc_annotations_dir = voc_dir / "Annotations"
    voc_image_sets_dir = voc_dir / "ImageSets" / "Main"

    assert voc_images_dir.exists()
    assert voc_annotations_dir.exists()
    assert voc_image_sets_dir.exists()

    for split in config.splits:
        split_ids = _read_voc_split_ids(voc_image_sets_dir, split)
        expected_images = _count_images(_split_path(fixture_root, config.images_dir, split))
        expected_annotations = _count_yolo_annotations(
            _split_path(fixture_root, config.labels_dir, split)
        )

        assert len(split_ids) == expected_images
        assert _count_voc_annotations(voc_annotations_dir, split_ids) == expected_annotations

    yolo_dir = tmp_path / "yolo"
    convert_dataset("voc", "yolo", voc_dir, yolo_dir, config)

    for split in config.splits:
        input_images_dir = _split_path(fixture_root, config.images_dir, split)
        input_labels_dir = _split_path(fixture_root, config.labels_dir, split)
        expected_images = _count_images(input_images_dir)
        expected_annotations = _count_yolo_annotations(input_labels_dir)

        yolo_images_dir = _split_path(yolo_dir, config.images_dir, split)
        yolo_labels_dir = _split_path(yolo_dir, config.labels_dir, split)

        assert yolo_images_dir.exists()
        assert yolo_labels_dir.exists()
        assert _count_images(yolo_images_dir) == expected_images
        assert _count_yolo_annotations(yolo_labels_dir) == expected_annotations

    yolo_report_dir = REPORTS_DIR / dataset_name / "voc_to_yolo"
    yolo_report_path = generate_fiftyone_report(
        dataset_format="yolo",
        output_dir=yolo_dir,
        config=config,
        report_dir=yolo_report_dir,
        split=config.splits[0],
    )
    assert yolo_report_path.exists()
