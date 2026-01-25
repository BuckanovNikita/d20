"""Command-line interface for d20 dataset conversion tool."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from d20.config import ConversionConfig, load_config
from d20.convert import convert_dataset

FORMATS = ("coco", "voc", "yolo")


def _parse_splits(value: str) -> list[str]:
    splits = [split.strip() for split in value.split(",") if split.strip()]
    return splits or ["data"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d20")
    parser.add_argument("source_format", choices=FORMATS)
    parser.add_argument("target_format", choices=FORMATS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits")
    parser.add_argument("--images-dir")
    parser.add_argument("--labels-dir")
    parser.add_argument("--annotations-dir")
    return parser


def _apply_overrides(config: ConversionConfig, args: argparse.Namespace) -> ConversionConfig:
    updates = {}
    if args.splits:
        updates["splits"] = _parse_splits(args.splits)
    if args.images_dir:
        updates["images_dir"] = args.images_dir
    if args.labels_dir:
        updates["labels_dir"] = args.labels_dir
    if args.annotations_dir:
        updates["annotations_dir"] = args.annotations_dir
    return config.model_copy(update=updates)


def main(argv: list[str] | None = None) -> None:
    """Run the d20 CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = _apply_overrides(load_config(config_path), args)

    logger.info(
        "Converting from {} to {}",
        args.source_format,
        args.target_format,
    )
    convert_dataset(
        input_format=args.source_format,
        output_format=args.target_format,
        input_path=Path(args.input),
        output_dir=Path(args.output),
        config=config,
    )
