"""FiftyOne report generation functionality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import fiftyone as fo
from loguru import logger

if TYPE_CHECKING:
    from d20.config import ConversionConfig


class UnsupportedDatasetFormatError(ValueError):
    """Raised when an unsupported dataset format is provided."""

    def __init__(self, dataset_format: str) -> None:
        """Initialize error with dataset format."""
        super().__init__(f"Unsupported dataset format: {dataset_format}")
        self.dataset_format = dataset_format


@dataclass
class ReportOptions:
    """Options for generating FiftyOne reports."""

    split: str | None = None
    max_samples: int = 10


def generate_fiftyone_report(
    dataset_format: str,
    output_dir: Path,
    config: ConversionConfig,
    report_dir: Path,
    options: ReportOptions | None = None,
) -> Path:
    """Generate a FiftyOne HTML report for a dataset."""
    if options is None:
        options = ReportOptions()

    dataset_format = dataset_format.lower()
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_name = options.split or (config.split_names[0] if config.split_names else "data")
    dataset_name = f"d20-{dataset_format}-{split_name}-{uuid4().hex}"

    dataset = _load_dataset(dataset_format, output_dir, config, split_name, dataset_name)
    try:
        drawn_dir = report_dir / "drawn"
        drawn_dir.mkdir(parents=True, exist_ok=True)
        view = dataset.limit(options.max_samples)
        drawn_paths = view.draw_labels(output_dir=str(drawn_dir), overwrite=True)
        html_path = report_dir / "index.html"
        html_path.write_text(_build_html(drawn_paths, report_dir))
        return html_path
    finally:
        logger.info("Cleaning FiftyOne dataset {}", dataset_name)
        dataset.delete()


def _load_dataset(
    dataset_format: str,
    output_dir: Path,
    config: ConversionConfig,
    split: str,
    dataset_name: str,
) -> fo.Dataset:  # type:ignore[name-defined]
    if dataset_format == "coco":
        data_path = _resolve_split_dir(output_dir / config.images_dir, split)
        labels_path = output_dir / config.annotations_dir / f"{split}.json"
        return fo.Dataset.from_dir(  # type: ignore[attr-defined]
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(data_path),
            labels_path=str(labels_path),
            name=dataset_name,
            overwrite=True,
        )

    if dataset_format == "yolo":
        return fo.Dataset.from_dir(  # type:ignore[attr-defined]
            dataset_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            name=dataset_name,
            overwrite=True,
        )

    if dataset_format == "voc":
        return fo.Dataset.from_dir(  # type:ignore[attr-defined]
            dataset_dir=str(output_dir),
            dataset_type=fo.types.VOCDetectionDataset,
            split=split,
            name=dataset_name,
            overwrite=True,
        )

    raise UnsupportedDatasetFormatError(dataset_format)


def _resolve_split_dir(base_dir: Path, split: str) -> Path:
    split_dir = base_dir / split
    return split_dir if split_dir.exists() else base_dir


def _build_html(drawn_paths: list[str], report_dir: Path) -> str:
    items = []
    for path in drawn_paths:
        resolved = Path(path)
        try:
            rel = resolved.relative_to(report_dir)
            src = rel.as_posix()
        except ValueError:
            src = resolved.as_posix()
        items.append(f'<div><img src="{src}" /></div>')

    body = "\n".join(items)
    return f'<html>\n<head><meta charset="utf-8"></head>\n<body>\n{body}\n</body>\n</html>'
