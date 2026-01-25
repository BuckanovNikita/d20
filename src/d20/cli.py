"""Command-line interface for d20 dataset conversion tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from typing_extensions import assert_never

from d20.convert import ConversionRequest, convert_dataset
from d20.formats import get_converter
from d20.reporting import export_fiftyone
from d20.types import (
    CocoDetectedParams,
    CocoWriteDetectedParams,
    ExportOptions,
    VocDetectedParams,
    VocWriteDetectedParams,
    WriteDetectedParams,
    YoloDetectedParams,
    YoloWriteDetectedParams,
)

FORMATS = ("coco", "voc", "yolo")


def _parse_list(value: str) -> list[str]:
    """Parse comma-separated or space-separated list."""
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _load_class_names_from_txt(txt_path: Path) -> list[str]:
    """Load class names from a text file (one per line)."""
    if not txt_path.exists():
        msg = f"Class names file not found: {txt_path}"
        raise FileNotFoundError(msg)
    lines = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
    if not lines:
        msg = "Class names file is empty"
        raise ValueError(msg)
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d20")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert command (explicit)
    convert_parser = subparsers.add_parser("convert", help="Convert dataset between formats")
    convert_parser.add_argument("source_format", choices=FORMATS, help="Source format (coco, voc, yolo)")
    convert_parser.add_argument("target_format", choices=FORMATS, help="Target format (coco, voc, yolo)")
    convert_parser.add_argument("--input", required=True, help="Input dataset path")
    convert_parser.add_argument("--output", required=True, help="Output directory path")
    convert_parser.add_argument("--class-names-file", help="Path to text file with class names (one per line)")
    convert_parser.add_argument("--splits", help="Comma-separated split names (default: auto-detect)")
    convert_parser.add_argument("--images-dir", help="Images directory name (default: auto-detect)")
    convert_parser.add_argument("--labels-dir", help="Labels directory name (default: auto-detect)")
    convert_parser.add_argument("--annotations-dir", help="Annotations directory name (default: auto-detect)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export dataset to FiftyOne App for visual inspection")
    export_parser.add_argument("format", choices=FORMATS, help="Dataset format (coco, voc, yolo)")
    export_parser.add_argument("--input", required=True, help="Input dataset path")
    export_parser.add_argument("--class-names-file", help="Path to text file with class names (one per line)")
    export_parser.add_argument("--splits", help="Comma-separated split names (default: auto-detect)")
    export_parser.add_argument("--images-dir", help="Images directory name (default: auto-detect)")
    export_parser.add_argument("--images-path", help="Path to images directory (for single JSON file input)")
    export_parser.add_argument("--labels-dir", help="Labels directory name (default: auto-detect)")
    export_parser.add_argument("--annotations-dir", help="Annotations directory name (default: auto-detect)")
    export_parser.add_argument("--split", help="Specific split to export (default: all splits)")

    # Backward compatibility: if first arg is a format, treat as convert command
    # This allows: d20 yolo coco --input ... (old format)
    parser.set_defaults(command=None)

    return parser


def _get_arg_value(args: argparse.Namespace, attr: str, default: str | None = None) -> str | None:
    """Get argument value if it exists and is set, otherwise return default."""
    if hasattr(args, attr):
        value = getattr(args, attr)
        return value if value else default
    return default


def _update_yolo_params(
    params: YoloDetectedParams,
    args: argparse.Namespace,
    class_names: list[str] | None = None,
    splits: list[str] | None = None,
) -> YoloDetectedParams:
    """Update YoloDetectedParams with CLI arguments."""
    return YoloDetectedParams(
        input_path=params.input_path,
        class_names=class_names if class_names is not None else params.class_names,
        splits=splits if splits is not None else params.splits,
        images_dir=_get_arg_value(args, "images_dir") or params.images_dir,
        labels_dir=_get_arg_value(args, "labels_dir") or params.labels_dir,
        annotations_dir=_get_arg_value(args, "annotations_dir") or params.annotations_dir,
        yaml_path=params.yaml_path,
        dataset_root=params.dataset_root,
    )


def _update_coco_params(
    params: CocoDetectedParams,
    args: argparse.Namespace,
    class_names: list[str] | None = None,
    splits: list[str] | None = None,
) -> CocoDetectedParams:
    """Update CocoDetectedParams with CLI arguments."""
    split_files = params.split_files
    input_path = params.input_path

    # Handle COCO single JSON file case
    if (
        hasattr(args, "images_path")
        and args.images_path
        and isinstance(params.input_path, Path)
        and params.input_path.is_file()
    ):
        images_path = Path(args.images_path)
        split_files = {params.input_path.stem: params.input_path}
        input_path = images_path if images_path.is_dir() else images_path.parent

    return CocoDetectedParams(
        input_path=input_path,
        class_names=class_names if class_names is not None else params.class_names,
        splits=splits if splits is not None else params.splits,
        images_dir=_get_arg_value(args, "images_dir") or params.images_dir,
        labels_dir=_get_arg_value(args, "labels_dir") or params.labels_dir,
        annotations_dir=_get_arg_value(args, "annotations_dir") or params.annotations_dir,
        split_files=split_files,
    )


def _update_voc_params(
    params: VocDetectedParams,
    args: argparse.Namespace,
    class_names: list[str] | None = None,
    splits: list[str] | None = None,
) -> VocDetectedParams:
    """Update VocDetectedParams with CLI arguments."""
    return VocDetectedParams(
        input_path=params.input_path,
        class_names=class_names if class_names is not None else params.class_names,
        splits=splits if splits is not None else params.splits,
        images_dir=_get_arg_value(args, "images_dir") or params.images_dir,
        labels_dir=_get_arg_value(args, "labels_dir") or params.labels_dir,
        annotations_dir=_get_arg_value(args, "annotations_dir") or params.annotations_dir,
        auto_detect_splits=params.auto_detect_splits,
    )


def _update_detected_params_with_cli(
    params: YoloDetectedParams | CocoDetectedParams | VocDetectedParams,
    args: argparse.Namespace,
) -> YoloDetectedParams | CocoDetectedParams | VocDetectedParams:
    """Update DetectedParams with CLI arguments (override autodetected values).

    Args:
        params: DetectedParams from autodetect()
        args: Parsed CLI arguments

    Returns:
        Updated DetectedParams

    """
    class_names: list[str] | None = None
    splits: list[str] | None = None

    # Override class names if provided
    if hasattr(args, "class_names_file") and args.class_names_file:
        class_names = _load_class_names_from_txt(Path(args.class_names_file))

    # Update splits if provided
    if hasattr(args, "splits") and args.splits:
        splits = _parse_list(args.splits)

    # Update params based on type
    if isinstance(params, YoloDetectedParams):
        return _update_yolo_params(params, args, class_names, splits)
    if isinstance(params, CocoDetectedParams):
        return _update_coco_params(params, args, class_names, splits)
    if isinstance(params, VocDetectedParams):
        return _update_voc_params(params, args, class_names, splits)

    assert_never(params)


def _build_write_params(
    format_name: str,
    detected_params: YoloDetectedParams | CocoDetectedParams | VocDetectedParams,
    args: argparse.Namespace,
) -> WriteDetectedParams:
    """Build WriteDetectedParams from DetectedParams and CLI arguments.

    Args:
        format_name: Output format name
        detected_params: DetectedParams from autodetect()
        args: Parsed CLI arguments

    Returns:
        WriteDetectedParams for the specified format

    """
    if not detected_params.class_names:
        msg = "Class names are required for writing datasets"
        raise ValueError(msg)

    class_names = detected_params.class_names
    splits = detected_params.splits
    images_dir = detected_params.images_dir or "images"
    labels_dir = detected_params.labels_dir or "labels"
    annotations_dir = detected_params.annotations_dir or "annotations"

    # Override with CLI args if provided
    if hasattr(args, "images_dir") and args.images_dir:
        images_dir = args.images_dir
    if hasattr(args, "labels_dir") and args.labels_dir:
        labels_dir = args.labels_dir
    if hasattr(args, "annotations_dir") and args.annotations_dir:
        annotations_dir = args.annotations_dir
    if hasattr(args, "splits") and args.splits:
        splits = _parse_list(args.splits)

    if format_name == "yolo":
        return YoloWriteDetectedParams(
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=annotations_dir,
        )
    if format_name == "coco":
        return CocoWriteDetectedParams(
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=annotations_dir,
        )
    if format_name == "voc":
        return VocWriteDetectedParams(
            class_names=class_names,
            splits=splits,
            images_dir=images_dir,
            labels_dir=labels_dir,
            annotations_dir=annotations_dir,
        )

    msg = f"Unknown format: {format_name}"
    raise ValueError(msg)


def _ensure_class_names(
    params: YoloDetectedParams | CocoDetectedParams | VocDetectedParams,
    args: argparse.Namespace,
) -> YoloDetectedParams | CocoDetectedParams | VocDetectedParams:
    """Ensure class names are available in params, loading from file if needed."""
    if params.class_names:
        return params

    if not (hasattr(args, "class_names_file") and args.class_names_file):
        msg = (
            "Class names must be provided. Use --class-names-file "
            "or ensure the dataset contains class information (YOLO YAML or COCO JSON)"
        )
        raise ValueError(msg)

    class_names = _load_class_names_from_txt(Path(args.class_names_file))
    if isinstance(params, YoloDetectedParams):
        return _update_yolo_params(params, args, class_names=class_names)
    if isinstance(params, CocoDetectedParams):
        return _update_coco_params(params, args, class_names=class_names)
    if isinstance(params, VocDetectedParams):
        return _update_voc_params(params, args, class_names=class_names)

    assert_never(params)


def _handle_convert_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the convert command."""
    if not hasattr(args, "source_format"):
        parser.error("Missing source and target formats")

    input_format = args.source_format.lower()
    output_format = args.target_format.lower()
    input_path = Path(args.input)

    # Autodetect input format parameters
    input_converter = get_converter(input_format)
    detected = input_converter.autodetect(input_path)

    # Type narrowing: autodetect returns specific implementation
    if not isinstance(detected, (YoloDetectedParams, CocoDetectedParams, VocDetectedParams)):
        msg = f"Unexpected DetectedParams type: {type(detected)}"
        raise TypeError(msg)

    # Update with CLI overrides
    read_params = _update_detected_params_with_cli(detected, args)
    read_params = _ensure_class_names(read_params, args)

    # Build write params
    write_params = _build_write_params(output_format, read_params, args)

    logger.info("Converting from {} to {}", input_format, output_format)

    request = ConversionRequest(
        input_format=input_format,
        output_format=output_format,
        input_path=input_path,
        output_dir=Path(args.output),
        read_params=read_params,
        write_params=write_params,
    )
    convert_dataset(request)


def _handle_export_command(args: argparse.Namespace) -> None:
    """Handle the export command."""
    format_name = args.format.lower()
    input_path_raw = Path(args.input)

    # Handle COCO single JSON file case
    input_path: Path | list[Path]
    if format_name == "coco" and input_path_raw.is_file() and input_path_raw.suffix.lower() == ".json":
        if hasattr(args, "images_path") and args.images_path:
            images_path = Path(args.images_path)
            input_path = images_path if images_path.is_dir() else images_path.parent
        else:
            input_path = [input_path_raw]
    else:
        input_path = input_path_raw

    # Autodetect format parameters
    converter = get_converter(format_name)
    detected = converter.autodetect(input_path)

    # Type narrowing: autodetect returns specific implementation
    if not isinstance(detected, (YoloDetectedParams, CocoDetectedParams, VocDetectedParams)):
        msg = f"Unexpected DetectedParams type: {type(detected)}"
        raise TypeError(msg)

    # Update with CLI overrides
    read_params = _update_detected_params_with_cli(detected, args)
    read_params = _ensure_class_names(read_params, args)

    # Build export options
    export_options = ExportOptions(split=getattr(args, "split", None))

    logger.info("Exporting {} dataset to FiftyOne App", format_name)
    export_fiftyone(read_params, export_options)


def main(argv: list[str] | None = None) -> None:
    """Run the d20 CLI."""
    parser = _build_parser()

    # Handle backward compatibility: if first arg is a format, treat as convert
    if argv is None:
        argv = sys.argv[1:]

    # Check if first argument is a format (backward compatibility)
    if argv and argv[0] in FORMATS and len(argv) > 1 and argv[1] in FORMATS:
        # Old format: d20 yolo coco --input ... --output ...
        # Insert "convert" as the command
        argv = ["convert", *argv]

    args = parser.parse_args(argv)

    if args.command == "convert" or (args.command is None and hasattr(args, "source_format")):
        _handle_convert_command(args, parser)
    elif args.command == "export":
        _handle_export_command(args)
    else:
        parser.error("Please specify a command: convert or export")
