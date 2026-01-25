"""Dataset conversion functionality."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from d20.formats import FormatNotRegisteredError, get_converter

if TYPE_CHECKING:
    from d20.config import ConversionConfig


class UnsupportedInputFormatError(ValueError):
    """Raised when an unsupported input format is provided."""

    def __init__(self, input_format: str) -> None:
        """Initialize error with input format."""
        super().__init__(f"Unsupported input format: {input_format}")
        self.input_format = input_format


class UnsupportedOutputFormatError(ValueError):
    """Raised when an unsupported output format is provided."""

    def __init__(self, output_format: str) -> None:
        """Initialize error with output format."""
        super().__init__(f"Unsupported output format: {output_format}")
        self.output_format = output_format


def convert_dataset(
    input_format: str,
    output_format: str,
    input_path: Path | list[Path],
    output_dir: Path,
    config: ConversionConfig,
    **read_options: Any,
) -> None:
    """Convert a dataset from one format to another.

    Args:
        input_format: Input format name (e.g., 'coco', 'voc', 'yolo')
        output_format: Output format name (e.g., 'coco', 'voc', 'yolo')
        input_path: Path to input dataset (directory, file, or list of paths)
        output_dir: Output directory path
        config: Conversion configuration
        **read_options: Additional options for reading (format-specific)

    """
    input_format = input_format.lower()
    output_format = output_format.lower()

    try:
        reader = get_converter(input_format)
    except FormatNotRegisteredError as e:
        raise UnsupportedInputFormatError(input_format) from e

    try:
        writer = get_converter(output_format)
    except FormatNotRegisteredError as e:
        raise UnsupportedOutputFormatError(output_format) from e

    input_path = Path(input_path) if not isinstance(input_path, list) else [Path(p) for p in input_path]
    output_dir = Path(output_dir)

    logger.info("Reading {} dataset from {}", input_format, input_path)
    splits = reader.read(input_path, config, **read_options)
    logger.info("Writing {} dataset to {}", output_format, output_dir)
    writer.write(output_dir, config, splits)
