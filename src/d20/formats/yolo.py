from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import Iterable

from loguru import logger
from pillow import Image

from d20.config import ConversionConfig
from d20.types import Annotation, DatasetSplit, ImageInfo

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
    )


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def _read_image_info(image_path: Path, images_dir: Path) -> ImageInfo:
    relative = image_path.relative_to(images_dir)
    with Image.open(image_path) as image:
        width, height = image.size

    return ImageInfo(
        image_id=image_path.stem,
        file_name=relative.as_posix(),
        width=width,
        height=height,
        path=image_path,
    )


def read_yolo_dataset(input_dir: Path, config: ConversionConfig) -> list[DatasetSplit]:
    splits: list[DatasetSplit] = []
    for split in config.split_names:
        images_dir = _resolve_split_dir(input_dir / config.images_dir, split)
        labels_dir = _resolve_split_dir(input_dir / config.labels_dir, split)

        if not images_dir.exists():
            logger.warning("YOLO images directory missing: {}", images_dir)
            splits.append(DatasetSplit(name=split, images=[], annotations=[]))
            continue

        images = [_read_image_info(path, images_dir) for path in _iter_images(images_dir)]
        annotations: list[Annotation] = []

        for image in images:
            label_path = labels_dir / Path(image.file_name).with_suffix(".txt")
            if not label_path.exists():
                continue

            for line in label_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    logger.warning("Skipping invalid YOLO label line: {}", line)
                    continue

                class_id = int(parts[0])
                if class_id >= len(config.class_names):
                    raise ValueError(f"Unknown class id: {class_id}")

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                bbox_x = (x_center - width / 2.0) * image.width
                bbox_y = (y_center - height / 2.0) * image.height
                bbox_w = width * image.width
                bbox_h = height * image.height

                annotations.append(
                    Annotation(
                        image_id=image.image_id,
                        category_id=class_id,
                        bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                    )
                )

        splits.append(DatasetSplit(name=split, images=images, annotations=annotations))

    return splits


def _write_data_yaml(output_dir: Path, config: ConversionConfig) -> None:
    data = {
        "path": ".",
        "names": config.class_names,
        "nc": len(config.class_names),
    }
    for split in config.split_names:
        if split == "data":
            data[split] = f"{config.images_dir}"
        else:
            data[split] = f"{config.images_dir}/{split}"

    output_dir.joinpath("data.yaml").write_text(
        yaml_safe_dump(data),
    )


def _group_annotations(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def write_yolo_dataset(
    output_dir: Path, config: ConversionConfig, splits: list[DatasetSplit]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        images_dir = output_dir / config.images_dir
        labels_dir = output_dir / config.labels_dir
        if split.name != "data":
            images_dir = images_dir / split.name
            labels_dir = labels_dir / split.name

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        grouped = _group_annotations(split.annotations)
        for image in split.images:
            relative = Path(image.file_name)
            target_image_path = images_dir / relative
            target_image_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(image.path, target_image_path)

            label_path = labels_dir / relative.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for annotation in grouped.get(image.image_id, []):
                x, y, w, h = annotation.bbox
                x_center = (x + w / 2.0) / image.width
                y_center = (y + h / 2.0) / image.height
                w_norm = w / image.width
                h_norm = h / image.height
                lines.append(
                    f"{annotation.category_id} "
                    f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )
            label_path.write_text("\n".join(lines))

    _write_data_yaml(output_dir, config)


def yaml_safe_dump(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False)
