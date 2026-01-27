"""Dataset conversion functionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from d20.formats import FormatNotRegisteredError, get_converter

if TYPE_CHECKING:
    from pathlib import Path

    from d20.types import DetectedParams, WriteDetectedParams


@dataclass
class ConversionRequest:
    """Request parameters for dataset conversion."""

    input_format: str
    output_format: str
    input_path: Path | list[Path]
    output_dir: Path
    read_params: DetectedParams
    write_params: WriteDetectedParams


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


def convert_dataset(request: ConversionRequest) -> None:
    """Convert a dataset from one format to another.

    Args:
        request: ConversionRequest with all conversion parameters

    """
    try:
        reader = get_converter(request.input_format)
    except FormatNotRegisteredError as e:
        raise UnsupportedInputFormatError(request.input_format) from e

    try:
        writer = get_converter(request.output_format)
    except FormatNotRegisteredError as e:
        raise UnsupportedOutputFormatError(request.output_format) from e

    logger.info(f"Reading {request.input_format} dataset from {request.input_path}")
    dataset = reader.read(request.read_params)
    logger.info(f"Writing {request.output_format} dataset to {request.output_dir}")
    writer.write(request.output_dir, dataset, request.write_params)
