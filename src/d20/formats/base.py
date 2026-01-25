"""Base classes and registry for format converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from d20.config import ConversionConfig
    from d20.types import DatasetSplit
else:
    pass


class UnknownFormatError(ValueError):
    """Raised when an unknown format is requested."""

    def __init__(self, format_name: str) -> None:
        """Initialize error with format name."""
        super().__init__(f"Unknown format: {format_name}")
        self.format_name = format_name


class FormatNotRegisteredError(ValueError):
    """Raised when a format converter is not registered."""

    def __init__(self, format_name: str) -> None:
        """Initialize error with format name."""
        super().__init__(f"Format converter not registered: {format_name}")
        self.format_name = format_name


class FormatConverter(ABC):
    """Base class for format converters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'coco', 'voc', 'yolo')."""
        ...

    @abstractmethod
    def read(
        self,
        input_path: Path | list[Path],
        config: ConversionConfig,
        **options: Any,
    ) -> list[DatasetSplit]:
        """Read a dataset from the specified path(s).

        Each converter can interpret input_path differently:
        - YOLO: if Path points to .yaml file, reads from it
        - COCO: if list[Path], reads each JSON as a separate split
        - VOC: always expects a directory

        Args:
            input_path: Path to dataset (directory, file, or list of paths)
            config: Conversion configuration
            **options: Format-specific options:
                - yolo: yaml_path (if not specified, looks for data.yaml in input_path)
                - coco: split_files (dict[str, Path] for explicit file mapping)
                - voc: auto_detect_splits (bool)

        Returns:
            List of dataset splits

        """
        ...

    @abstractmethod
    def write(
        self,
        output_dir: Path,
        config: ConversionConfig,
        splits: list[DatasetSplit],
        **options: Any,
    ) -> None:
        """Write a dataset to the specified directory.

        Args:
            output_dir: Output directory path
            config: Conversion configuration
            splits: List of dataset splits to write
            **options: Format-specific options

        """
        ...


class ConverterRegistry:
    """Registry for format converters."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._converters: dict[str, type[FormatConverter]] = {}

    def register(self, format_name: str, converter_class: type[FormatConverter]) -> None:
        """Register a format converter.

        Args:
            format_name: Name of the format (e.g., 'coco', 'voc', 'yolo')
            converter_class: Converter class to register

        """
        self._converters[format_name.lower()] = converter_class

    def get(self, format_name: str) -> FormatConverter:
        """Get a converter instance for the specified format.

        Args:
            format_name: Name of the format

        Returns:
            Converter instance

        Raises:
            FormatNotRegisteredError: If format is not registered

        """
        format_name_lower = format_name.lower()
        if format_name_lower not in self._converters:
            raise FormatNotRegisteredError(format_name_lower)
        converter_class = self._converters[format_name_lower]
        return converter_class()

    def list_formats(self) -> list[str]:
        """List all registered format names.

        Returns:
            List of registered format names

        """
        return sorted(self._converters.keys())

    def is_registered(self, format_name: str) -> bool:
        """Check if a format is registered.

        Args:
            format_name: Name of the format

        Returns:
            True if format is registered, False otherwise

        """
        return format_name.lower() in self._converters


# Global registry instance
_registry = ConverterRegistry()


def register_converter(format_name: str) -> Any:
    """Register a format converter.

    Args:
        format_name: Name of the format to register

    Returns:
        Decorator function

    Example:
        @register_converter("coco")
        class CocoConverter(FormatConverter):
            ...

    """

    def decorator(cls: type[FormatConverter]) -> type[FormatConverter]:
        _registry.register(format_name, cls)
        return cls

    return decorator


def get_converter(format_name: str) -> FormatConverter:
    """Get a converter instance for the specified format.

    Args:
        format_name: Name of the format

    Returns:
        Converter instance

    Raises:
        FormatNotRegisteredError: If format is not registered

    """
    return _registry.get(format_name)


def list_formats() -> list[str]:
    """List all registered format names.

    Returns:
        List of registered format names

    """
    return _registry.list_formats()
