"""VOC format dataset reading and writing."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, ElementTree, SubElement

from defusedxml.ElementTree import parse as defused_parse
from loguru import logger
from PIL import Image
from typing_extensions import override

from d20.formats.base import FormatConverter, register_converter
from d20.formats.utils import group_annotations_by_image
from d20.types import (
    Annotation,
    Dataset,
    ImageInfo,
    Split,
    VocDetectedParams,
    VocWriteDetectedParams,
)

if TYPE_CHECKING:
    from xml.etree.ElementTree import ElementTree as _ElementTree


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


class InvalidVocNumericValueError(ValueError):
    """Raised when VOC XML contains invalid numeric value."""

    def __init__(self, field: str, value: str, error: Exception) -> None:
        """Initialize error with field, value and original error."""
        super().__init__(f"Invalid numeric value in VOC XML: field '{field}' = '{value}': {error}")
        self.field = field
        self.value = value
        self.original_error = error


class InvalidVocBboxError(ValueError):
    """Raised when VOC bounding box is invalid."""

    def __init__(self, bbox: tuple[int, int, int, int], image_size: tuple[int, int] | None = None) -> None:
        """Initialize error with bbox and optional image size."""
        xmin, ymin, xmax, ymax = bbox
        if image_size:
            img_w, img_h = image_size
            msg = (
                f"Invalid bounding box ({xmin}, {ymin}, {xmax}, {ymax}) for image size ({img_w}, {img_h}): "
                f"xmin={xmin} < xmax={xmax} and ymin={ymin} < ymax={ymax} must be true, "
                f"bbox must be within image bounds and non-negative"
            )
        else:
            msg = (
                f"Invalid bounding box ({xmin}, {ymin}, {xmax}, {ymax}): "
                f"xmin={xmin} < xmax={xmax} and ymin={ymin} < ymax={ymax} must be true, "
                f"coordinates must be non-negative"
            )
        super().__init__(msg)
        self.bbox = bbox
        self.image_size = image_size


def _read_split_ids(image_sets_dir: Path, split: str) -> list[str]:
    split_path = image_sets_dir / f"{split}.txt"
    if not split_path.exists():
        logger.warning(f"VOC split file missing: {split_path}")
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

    def autodetect(
        self,
        input_path: Path | list[Path],
    ) -> VocDetectedParams:
        """Autodetect VOC dataset parameters without reading the full dataset.

        Args:
            input_path: Path to dataset directory

        Returns:
            VocDetectedParams with detected parameters

        """
        # Normalize input path
        if isinstance(input_path, list):
            if len(input_path) > 0:
                input_path = input_path[0]
            else:
                raise EmptyPathListError

        input_dir = Path(input_path)
        voc_image_sets_dir = input_dir / "ImageSets" / "Main"

        # Auto-detect splits if ImageSets/Main exists
        splits: list[str] | None = None
        auto_detect_splits = False
        if voc_image_sets_dir.exists():
            detected_splits = self._detect_splits(voc_image_sets_dir)
            if detected_splits:
                splits = detected_splits
                auto_detect_splits = True

        # VOC has fixed directory structure
        images_dir: str | None = "JPEGImages"
        labels_dir: str | None = None  # VOC doesn't use labels_dir
        annotations_dir: str | None = "Annotations"

        # Class names cannot be auto-detected from VOC format
        class_names: list[str] | None = None

        return VocDetectedParams(
            input_path=input_path,
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=annotations_dir,
            auto_detect_splits=auto_detect_splits,
        )

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
            logger.warning(f"Empty XML root in VOC: {xml_path}")
            return None

        filename = root.findtext("filename")
        if not filename:
            logger.warning(f"Missing filename in VOC: {xml_path}")
            return None

        image_path = voc_images_dir / filename
        size_node = root.find("size")
        if size_node is not None:
            width_str = size_node.findtext("width", "0")
            height_str = size_node.findtext("height", "0")
            try:
                width = int(width_str)
            except ValueError as e:
                raise InvalidVocNumericValueError("width", width_str, e) from e
            try:
                height = int(height_str)
            except ValueError as e:
                raise InvalidVocNumericValueError("height", height_str, e) from e

            if width <= 0 or height <= 0:
                raise InvalidVocBboxError((0, 0, width, height), (width, height))
        else:
            width, height = _read_image_size(image_path)

        return ImageInfo(
            image_id=image_id,
            file_name=filename,
            width=width,
            height=height,
            path=image_path,
        )

    def _parse_bbox_coordinates(
        self,
        bbox: Element,
    ) -> tuple[int, int, int, int]:
        """Parse and validate bounding box coordinates from VOC XML.

        Args:
            bbox: XML bndbox element

        Returns:
            Tuple of (xmin, ymin, xmax, ymax)

        Raises:
            InvalidVocNumericValueError: If coordinate conversion fails
            InvalidVocBboxError: If bbox coordinates are invalid

        """
        xmin_str = bbox.findtext("xmin", "1")
        ymin_str = bbox.findtext("ymin", "1")
        xmax_str = bbox.findtext("xmax", "1")
        ymax_str = bbox.findtext("ymax", "1")

        try:
            xmin = int(xmin_str) - 1
        except ValueError as e:
            raise InvalidVocNumericValueError("xmin", xmin_str, e) from e

        try:
            ymin = int(ymin_str) - 1
        except ValueError as e:
            raise InvalidVocNumericValueError("ymin", ymin_str, e) from e

        try:
            xmax = int(xmax_str)
        except ValueError as e:
            raise InvalidVocNumericValueError("xmax", xmax_str, e) from e

        try:
            ymax = int(ymax_str)
        except ValueError as e:
            raise InvalidVocNumericValueError("ymax", ymax_str, e) from e

        # Validate bbox coordinates
        if xmin < 0 or ymin < 0:
            raise InvalidVocBboxError((xmin, ymin, xmax, ymax))
        if xmin >= xmax or ymin >= ymax:
            raise InvalidVocBboxError((xmin, ymin, xmax, ymax))

        return (xmin, ymin, xmax, ymax)

    def _parse_annotations_from_xml(
        self,
        root: Element,
        image_id: str,
        class_names: list[str],
    ) -> list[Annotation]:
        """Parse annotations from VOC XML root element.

        Args:
            root: XML root element
            image_id: Image identifier
            class_names: List of class names

        Returns:
            List of annotations

        Raises:
            UnknownClassNameError: If unknown class name is found

        """
        annotations: list[Annotation] = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if not name:
                logger.warning(f"Missing 'name' field in VOC object for image {image_id}, skipping")
                continue
            if name not in class_names:
                raise UnknownClassNameError(name)

            bbox = obj.find("bndbox")
            if bbox is None:
                logger.warning(f"Missing 'bndbox' element in VOC object for image {image_id}, skipping")
                continue

            xmin, ymin, xmax, ymax = self._parse_bbox_coordinates(bbox)

            annotations.append(
                Annotation(
                    image_id=image_id,
                    category_id=class_names.index(name),
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
        class_names: list[str],
    ) -> Split:
        """Read a single split from VOC dataset.

        Args:
            split_name: Name of the split
            voc_image_sets_dir: Directory with split files
            voc_annotations_dir: Directory with XML annotations
            voc_images_dir: Directory with images
            class_names: List of class names

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
                split_annotations = self._parse_annotations_from_xml(root, image_id, class_names)
                annotations.extend(split_annotations)

        return Split(name=split_name, images=images, annotations=annotations)

    @override
    def read(
        self,
        params: VocDetectedParams,  # type: ignore[override]
    ) -> Dataset:
        """Read a VOC format dataset from disk.

        Args:
            params: VocDetectedParams with all necessary information

        Returns:
            Dataset containing all splits

        """
        if not params.class_names:
            msg = "Class names are required for VOC format"
            raise ValueError(msg)

        if isinstance(params.input_path, list):
            if len(params.input_path) > 0:
                input_dir = Path(params.input_path[0])
            else:
                raise EmptyPathListError
        else:
            input_dir = Path(params.input_path)

        voc_images_dir = input_dir / "JPEGImages"
        voc_annotations_dir = input_dir / "Annotations"
        voc_image_sets_dir = input_dir / "ImageSets" / "Main"

        # Determine splits
        if params.auto_detect_splits and voc_image_sets_dir.exists():
            splits = self._detect_splits(voc_image_sets_dir)
        else:
            splits = params.splits or ["data"]

        splits_dict: dict[str, Split] = {}
        for split_name in splits:
            split = self._read_split(
                split_name,
                voc_image_sets_dir,
                voc_annotations_dir,
                voc_images_dir,
                params.class_names,
            )
            splits_dict[split_name] = split

        return Dataset(splits=splits_dict, class_names=params.class_names)

    @override
    def write(
        self,
        output_dir: Path,
        dataset: Dataset,
        params: VocWriteDetectedParams,  # type: ignore[override]
    ) -> None:
        """Write a dataset in VOC format to disk.

        Args:
            output_dir: Output directory path
            dataset: Dataset containing all splits
            params: VocWriteDetectedParams with write configuration

        """
        voc_images_dir = output_dir / "JPEGImages"
        voc_annotations_dir = output_dir / "Annotations"
        voc_image_sets_dir = output_dir / "ImageSets" / "Main"

        voc_images_dir.mkdir(parents=True, exist_ok=True)
        voc_annotations_dir.mkdir(parents=True, exist_ok=True)
        voc_image_sets_dir.mkdir(parents=True, exist_ok=True)

        splits_to_write = params.splits or list(dataset.splits.keys())
        for split_name in splits_to_write:
            if split_name not in dataset.splits:
                logger.warning(f"Split '{split_name}' not found in dataset, skipping")
                continue

            split = dataset.splits[split_name]
            grouped = group_annotations_by_image(split.annotations)
            split_ids: list[str] = []

            for image in split.images:
                image_id = Path(image.file_name).stem
                split_ids.append(image_id)
                copy2(image.path, voc_images_dir / image.file_name)

                annotations = grouped.get(image.image_id, [])
                tree = _create_annotation_xml(image, annotations, params.class_names)
                xml_path = voc_annotations_dir / f"{image_id}.xml"
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            split_path = voc_image_sets_dir / f"{split.name}.txt"
            split_path.write_text("\n".join(split_ids))
