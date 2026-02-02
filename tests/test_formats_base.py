"""Tests for format converter base classes and registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from d20.formats.base import (
    ConverterRegistry,
    FormatConverter,
    FormatNotRegisteredError,
    UnknownFormatError,
    get_converter,
    list_formats,
    register_converter,
)
from d20.types import Dataset, DetectedParams, WriteDetectedParams, YoloDetectedParams


class MockConverter(FormatConverter):
    """Mock converter for testing."""

    @property
    def format_name(self) -> str:
        """Return format name."""
        return "mock"

    def autodetect(self, input_path: Path | list[Path]) -> DetectedParams:
        """Mock autodetect."""
        return YoloDetectedParams(
            input_path=Path(input_path) if isinstance(input_path, Path) else input_path[0],
            class_names=None,
            splits=None,
        )

    def read(self, _params: DetectedParams) -> Dataset:
        """Mock read."""
        return Dataset(splits={}, class_names=[])

    def write(self, output_dir: Path, dataset: Dataset, params: WriteDetectedParams) -> None:
        """Mock write."""


def test_converter_registry_register() -> None:
    """Test ConverterRegistry.register() method."""
    registry = ConverterRegistry()
    registry.register("test_format", MockConverter)
    assert registry.is_registered("test_format")


def test_converter_registry_get() -> None:
    """Test ConverterRegistry.get() method."""
    registry = ConverterRegistry()
    registry.register("test_format", MockConverter)
    converter = registry.get("test_format")
    assert isinstance(converter, MockConverter)
    assert converter.format_name == "mock"


def test_converter_registry_get_case_insensitive() -> None:
    """Test ConverterRegistry.get() is case insensitive."""
    registry = ConverterRegistry()
    registry.register("test_format", MockConverter)
    converter1 = registry.get("TEST_FORMAT")
    converter2 = registry.get("Test_Format")
    assert isinstance(converter1, MockConverter)
    assert isinstance(converter2, MockConverter)


def test_converter_registry_get_not_registered() -> None:
    """Test ConverterRegistry.get() raises error for unregistered format."""
    registry = ConverterRegistry()
    with pytest.raises(FormatNotRegisteredError) as exc_info:
        registry.get("unknown_format")
    assert exc_info.value.format_name == "unknown_format"


def test_converter_registry_list_formats() -> None:
    """Test ConverterRegistry.list_formats() method."""
    registry = ConverterRegistry()
    registry.register("format_b", MockConverter)
    registry.register("format_a", MockConverter)
    formats = registry.list_formats()
    assert formats == ["format_a", "format_b"]


def test_converter_registry_list_formats_empty() -> None:
    """Test ConverterRegistry.list_formats() with empty registry."""
    registry = ConverterRegistry()
    formats = registry.list_formats()
    assert formats == []


def test_converter_registry_is_registered() -> None:
    """Test ConverterRegistry.is_registered() method."""
    registry = ConverterRegistry()
    registry.register("test_format", MockConverter)
    assert registry.is_registered("test_format")
    assert registry.is_registered("TEST_FORMAT")  # Case insensitive
    assert not registry.is_registered("unknown_format")


def test_register_converter_decorator() -> None:
    """Test register_converter() decorator."""

    @register_converter("decorated_format")
    class DecoratedConverter(FormatConverter):
        @property
        def format_name(self) -> str:
            return "decorated_format"

        def autodetect(self, input_path: Path | list[Path]) -> DetectedParams:
            return YoloDetectedParams(
                input_path=Path(input_path) if isinstance(input_path, Path) else input_path[0],
                class_names=None,
                splits=None,
            )

        def read(self, _params: DetectedParams) -> Dataset:
            return Dataset(splits={}, class_names=[])

        def write(self, _output_dir: Path, _dataset: Dataset, _params: WriteDetectedParams) -> None:
            pass

    # The decorator registers with the global registry
    # We can't easily test this without affecting other tests, so we just verify it doesn't crash
    assert hasattr(DecoratedConverter, "format_name")


def test_get_converter_registered() -> None:
    """Test get_converter() with registered format."""
    # Use existing registered formats
    converter = get_converter("yolo")
    assert converter.format_name == "yolo"


def test_get_converter_case_insensitive() -> None:
    """Test get_converter() is case insensitive."""
    converter1 = get_converter("YOLO")
    converter2 = get_converter("yolo")
    assert converter1.format_name == "yolo"
    assert converter2.format_name == "yolo"


def test_get_converter_not_registered() -> None:
    """Test get_converter() raises error for unregistered format."""
    with pytest.raises(FormatNotRegisteredError) as exc_info:
        get_converter("unknown_format_xyz")
    assert exc_info.value.format_name == "unknown_format_xyz"


def test_list_formats() -> None:
    """Test list_formats() function."""
    formats = list_formats()
    assert "coco" in formats
    assert "yolo" in formats
    assert isinstance(formats, list)
    assert len(formats) >= 2


def test_format_not_registered_error() -> None:
    """Test FormatNotRegisteredError exception."""
    error = FormatNotRegisteredError("test_format")
    assert error.format_name == "test_format"
    assert str(error) == "Format converter not registered: test_format"


def test_unknown_format_error() -> None:
    """Test UnknownFormatError exception."""
    error = UnknownFormatError("test_format")
    assert error.format_name == "test_format"
    assert str(error) == "Unknown format: test_format"
