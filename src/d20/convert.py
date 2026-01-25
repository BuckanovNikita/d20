from __future__ import annotations

from pathlib import Path

from loguru import logger

from d20.config import ConversionConfig
from d20.formats.coco import read_coco_dataset, write_coco_dataset
from d20.formats.voc import read_voc_dataset, write_voc_dataset
from d20.formats.yolo import read_yolo_dataset, write_yolo_dataset

READERS = {
    "coco": read_coco_dataset,
    "voc": read_voc_dataset,
    "yolo": read_yolo_dataset,
}

WRITERS = {
    "coco": write_coco_dataset,
    "voc": write_voc_dataset,
    "yolo": write_yolo_dataset,
}


def convert_dataset(
    input_format: str,
    output_format: str,
    input_dir: Path,
    output_dir: Path,
    config: ConversionConfig,
) -> None:
    input_format = input_format.lower()
    output_format = output_format.lower()

    if input_format not in READERS:
        raise ValueError(f"Unsupported input format: {input_format}")
    if output_format not in WRITERS:
        raise ValueError(f"Unsupported output format: {output_format}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    logger.info("Reading {} dataset from {}", input_format, input_dir)
    splits = READERS[input_format](input_dir, config)
    logger.info("Writing {} dataset to {}", output_format, output_dir)
    WRITERS[output_format](output_dir, config, splits)
