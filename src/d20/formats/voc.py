"""VOC format dataset reading and writing."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element, ElementTree, SubElement

from defusedxml.ElementTree import parse as defused_parse
from loguru import logger
from PIL import Image

from d20.formats.base import FormatConverter, register_converter
from d20.types import Annotation, DatasetSplit, ImageInfo

if TYPE_CHECKING:
    from xml.etree.ElementTree import ElementTree as _ElementTree

from d20.config import ConversionConfig


class UnknownClassNameError(ValueError):
    """Raised when an unknown class name is encountered."""

    def __init__(self, name: str) -> None:
        """Initialize error with class name."""
        super().__init__(f"Unknown class name: {name}")
        self.name = name


class EmptyPathListError(ValueError):
    """Raised when empty path list is provided for VOC format."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__("Empty path list provided for VOC format")


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


def _group_annotations(annotations: list[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def _create_annotation_xml(
    image: ImageInfo,
    annotations: list[Annotation],
    class_names: list[str],
) -> _ElementTree:
    root = Element("annotation")
    SubElement(root, "folder").text = "JPEGImages"
    SubElement(root, "filename").text = image.file_name

    size = SubElement(root, "size")
    SubElement(size, "width").text = str(image.width)
    SubElement(size, "height").text = str(image.height)
    SubElement(size, "depth").text = "3"

    for annotation in annotations:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = class_names[annotation.category_id]
        bbox = SubElement(obj, "bndbox")

        x, y, w, h = annotation.bbox
        xmin = round(x) + 1
        ymin = round(y) + 1
        xmax = round(x + w)
        ymax = round(y + h)

        SubElement(bbox, "xmin").text = str(xmin)
        SubElement(bbox, "ymin").text = str(ymin)
        SubElement(bbox, "xmax").text = str(xmax)
        SubElement(bbox, "ymax").text = str(ymax)

    return ElementTree(root)


@register_converter("voc")
class VocConverter(FormatConverter):
    """Converter for PASCAL VOC format datasets."""

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "voc"

    def _detect_splits(self, image_sets_dir: Path) -> list[str]:
        """Auto-detect splits from ImageSets/Main directory.

        Args:
            image_sets_dir: Path to ImageSets/Main directory

        Returns:
            List of split names

        """
        splits = []
        for split_file in image_sets_dir.glob("*.txt"):
            split_name = split_file.stem
            if split_name:
                splits.append(split_name)
        return splits if splits else ["data"]

    def _normalize_input_path(self, input_path: Path | list[Path]) -> Path:
        """Normalize input path for VOC format.

        Args:
            input_path: Input path (directory or list)

        Returns:
            Normalized directory path

        Raises:
            EmptyPathListError: If empty list is provided

        """
        if isinstance(input_path, list):
            if len(input_path) > 0:
                return Path(input_path[0])
            raise EmptyPathListError
        return Path(input_path)

    def _prepare_config(
        self,
        config: ConversionConfig,
        voc_image_sets_dir: Path,
        **options: Any,
    ) -> ConversionConfig:
        """Prepare configuration with auto-detected splits if requested.

        Args:
            config: Original conversion configuration
            voc_image_sets_dir: Path to ImageSets/Main directory
            **options: Additional options

        Returns:
            Updated configuration

        """
        if options.get("auto_detect_splits", False):
            detected_splits = self._detect_splits(voc_image_sets_dir)
            if detected_splits:
                return ConversionConfig(
                    class_names=config.class_names,
                    splits=detected_splits,
                    images_dir=config.images_dir,
                    labels_dir=config.labels_dir,
                    annotations_dir=config.annotations_dir,
                )
        return config

    def _parse_image_xml(
        self,
        xml_path: Path,
        image_id: str,
        voc_images_dir: Path,
    ) -> ImageInfo | None:
        """Parse image information from VOC XML file.

        Args:
            xml_path: Path to XML annotation file
            image_id: Image identifier
            voc_images_dir: Directory with images

        Returns:
            ImageInfo if successful, None otherwise

        """
        tree = defused_parse(xml_path)
        root = tree.getroot()
        if root is None:
            logger.warning("Empty XML root in VOC: {}", xml_path)
            return None

        filename = root.findtext("filename")
        if not filename:
            logger.warning("Missing filename in VOC: {}", xml_path)
            return None

        image_path = voc_images_dir / filename
        size_node = root.find("size")
        if size_node is not None:
            width = int(size_node.findtext("width", "0"))
            height = int(size_node.findtext("height", "0"))
        else:
            width, height = _read_image_size(image_path)

        return ImageInfo(
            image_id=image_id,
            file_name=filename,
            width=width,
            height=height,
            path=image_path,
        )

    def _parse_annotations_from_xml(
        self,
        root: Any,
        image_id: str,
        config: ConversionConfig,
    ) -> list[Annotation]:
        """Parse annotations from VOC XML root element.

        Args:
            root: XML root element
            image_id: Image identifier
            config: Conversion configuration

        Returns:
            List of annotations

        Raises:
            UnknownClassNameError: If unknown class name is found

        """
        annotations: list[Annotation] = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if not name:
                continue
            if name not in config.class_names:
                raise UnknownClassNameError(name)

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
                ),
            )
        return annotations

    def _read_split(
        self,
        split_name: str,
        voc_image_sets_dir: Path,
        voc_annotations_dir: Path,
        voc_images_dir: Path,
        config: ConversionConfig,
    ) -> DatasetSplit:
        """Read a single split from VOC dataset.

        Args:
            split_name: Name of the split
            voc_image_sets_dir: Directory with split files
            voc_annotations_dir: Directory with XML annotations
            voc_images_dir: Directory with images
            config: Conversion configuration

        Returns:
            Dataset split

        """
        split_ids = _read_split_ids(voc_image_sets_dir, split_name)
        images: list[ImageInfo] = []
        annotations: list[Annotation] = []

        for image_id in split_ids:
            xml_path = voc_annotations_dir / f"{image_id}.xml"
            image_info = self._parse_image_xml(xml_path, image_id, voc_images_dir)
            if image_info is None:
                continue

            images.append(image_info)

            tree = defused_parse(xml_path)
            root = tree.getroot()
            if root is not None:
                split_annotations = self._parse_annotations_from_xml(root, image_id, config)
                annotations.extend(split_annotations)

        return DatasetSplit(name=split_name, images=images, annotations=annotations)

    def read(
        self,
        input_path: Path | list[Path],
        config: ConversionConfig,
        **options: Any,
    ) -> list[DatasetSplit]:
        """Read a VOC format dataset from disk.

        Args:
            input_path: Path to dataset directory (VOC doesn't support list[Path])
            config: Conversion configuration
            **options: Additional options:
                - auto_detect_splits: bool - automatically detect splits from ImageSets/Main

        Returns:
            List of dataset splits

        """
        input_dir = self._normalize_input_path(input_path)
        voc_images_dir = input_dir / "JPEGImages"
        voc_annotations_dir = input_dir / "Annotations"
        voc_image_sets_dir = input_dir / "ImageSets" / "Main"

        config = self._prepare_config(config, voc_image_sets_dir, **options)

        splits: list[DatasetSplit] = []
        for split_name in config.split_names:
            split = self._read_split(
                split_name,
                voc_image_sets_dir,
                voc_annotations_dir,
                voc_images_dir,
                config,
            )
            splits.append(split)

        return splits

    def write(
        self,
        output_dir: Path,
        config: ConversionConfig,
        splits: list[DatasetSplit],
        **_options: Any,
    ) -> None:
        """Write a dataset in VOC format to disk.

        Args:
            output_dir: Output directory path
            config: Conversion configuration
            splits: List of dataset splits to write
            **options: Additional options (unused)

        """
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
