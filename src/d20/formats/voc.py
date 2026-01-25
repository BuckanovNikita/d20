from __future__ import annotations

from pathlib import Path
from shutil import copy2

from defusedxml import ElementTree as ET
from loguru import logger
from pillow import Image

from d20.config import ConversionConfig
from d20.types import Annotation, DatasetSplit, ImageInfo


def _read_split_ids(image_sets_dir: Path, split: str) -> list[str]:
    split_path = image_sets_dir / f"{split}.txt"
    if not split_path.exists():
        logger.warning("VOC split file missing: {}", split_path)
        return []

    ids = []
    for line in split_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        ids.append(stripped.split()[0])
    return ids


def _read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return width, height


def read_voc_dataset(input_dir: Path, config: ConversionConfig) -> list[DatasetSplit]:
    voc_images_dir = input_dir / "JPEGImages"
    voc_annotations_dir = input_dir / "Annotations"
    voc_image_sets_dir = input_dir / "ImageSets" / "Main"

    splits: list[DatasetSplit] = []
    for split in config.split_names:
        split_ids = _read_split_ids(voc_image_sets_dir, split)
        images: list[ImageInfo] = []
        annotations: list[Annotation] = []

        for image_id in split_ids:
            xml_path = voc_annotations_dir / f"{image_id}.xml"
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.findtext("filename")
            if not filename:
                logger.warning("Missing filename in VOC: {}", xml_path)
                continue

            image_path = voc_images_dir / filename
            size_node = root.find("size")
            if size_node is not None:
                width = int(size_node.findtext("width", "0"))
                height = int(size_node.findtext("height", "0"))
            else:
                width, height = _read_image_size(image_path)

            images.append(
                ImageInfo(
                    image_id=image_id,
                    file_name=filename,
                    width=width,
                    height=height,
                    path=image_path,
                )
            )

            for obj in root.findall("object"):
                name = obj.findtext("name")
                if not name:
                    continue
                if name not in config.class_names:
                    raise ValueError(f"Unknown class name: {name}")

                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                xmin = int(bbox.findtext("xmin", "1")) - 1
                ymin = int(bbox.findtext("ymin", "1")) - 1
                xmax = int(bbox.findtext("xmax", "1"))
                ymax = int(bbox.findtext("ymax", "1"))

                annotations.append(
                    Annotation(
                        image_id=image_id,
                        category_id=config.class_names.index(name),
                        bbox=(xmin, ymin, xmax - xmin, ymax - ymin),
                    )
                )

        splits.append(DatasetSplit(name=split, images=images, annotations=annotations))

    return splits


def _group_annotations(annotations: list[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def _create_annotation_xml(
    image: ImageInfo,
    annotations: list[Annotation],
    class_names: list[str],
) -> ET.ElementTree:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = image.file_name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.width)
    ET.SubElement(size, "height").text = str(image.height)
    ET.SubElement(size, "depth").text = "3"

    for annotation in annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[annotation.category_id]
        bbox = ET.SubElement(obj, "bndbox")

        x, y, w, h = annotation.bbox
        xmin = int(round(x)) + 1
        ymin = int(round(y)) + 1
        xmax = int(round(x + w))
        ymax = int(round(y + h))

        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    return ET.ElementTree(root)


def write_voc_dataset(
    output_dir: Path, config: ConversionConfig, splits: list[DatasetSplit]
) -> None:
    voc_images_dir = output_dir / "JPEGImages"
    voc_annotations_dir = output_dir / "Annotations"
    voc_image_sets_dir = output_dir / "ImageSets" / "Main"

    voc_images_dir.mkdir(parents=True, exist_ok=True)
    voc_annotations_dir.mkdir(parents=True, exist_ok=True)
    voc_image_sets_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        grouped = _group_annotations(split.annotations)
        split_ids: list[str] = []

        for image in split.images:
            image_id = Path(image.file_name).stem
            split_ids.append(image_id)
            copy2(image.path, voc_images_dir / image.file_name)

            annotations = grouped.get(image.image_id, [])
            tree = _create_annotation_xml(image, annotations, config.class_names)
            xml_path = voc_annotations_dir / f"{image_id}.xml"
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        split_path = voc_image_sets_dir / f"{split.name}.txt"
        split_path.write_text("\n".join(split_ids))
