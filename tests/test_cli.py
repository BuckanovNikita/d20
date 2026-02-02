"""Tests for CLI module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from d20.cli import (
    _build_parser,
    _build_write_params,
    _ensure_class_names,
    _get_arg_value,
    _handle_convert_command,
    _handle_export_command,
    _load_class_names_from_txt,
    _parse_list,
    _update_detected_params_with_cli,
    main,
)
from d20.types import (
    CocoDetectedParams,
    CocoWriteDetectedParams,
    YoloDetectedParams,
    YoloWriteDetectedParams,
)


def test_parse_list_comma_separated() -> None:
    """Test _parse_list() with comma-separated values."""
    result = _parse_list("train,val,test")
    assert result == ["train", "val", "test"]


def test_parse_list_space_separated() -> None:
    """Test _parse_list() with space-separated values."""
    result = _parse_list("train val test")
    assert result == ["train", "val", "test"]


def test_parse_list_mixed() -> None:
    """Test _parse_list() with mixed separators."""
    result = _parse_list("train, val , test")
    assert result == ["train", "val", "test"]


def test_parse_list_empty() -> None:
    """Test _parse_list() with empty string."""
    result = _parse_list("")
    assert result == []


def test_load_class_names_from_txt_valid(tmp_path: Path) -> None:
    """Test _load_class_names_from_txt() with valid file."""
    txt_file = tmp_path / "classes.txt"
    txt_file.write_text("cat\ndog\nbird\n")

    names = _load_class_names_from_txt(txt_file)
    assert names == ["cat", "dog", "bird"]


def test_load_class_names_from_txt_with_empty_lines(tmp_path: Path) -> None:
    """Test _load_class_names_from_txt() handles empty lines."""
    txt_file = tmp_path / "classes.txt"
    txt_file.write_text("cat\n\ndog\n  \nbird\n")

    names = _load_class_names_from_txt(txt_file)
    assert names == ["cat", "dog", "bird"]


def test_load_class_names_from_txt_missing_file() -> None:
    """Test _load_class_names_from_txt() raises error for missing file."""
    txt_file = Path("/nonexistent/classes.txt")
    with pytest.raises(FileNotFoundError):
        _load_class_names_from_txt(txt_file)


def test_load_class_names_from_txt_empty_file(tmp_path: Path) -> None:
    """Test _load_class_names_from_txt() raises error for empty file."""
    txt_file = tmp_path / "classes.txt"
    txt_file.write_text("")

    with pytest.raises(ValueError, match="Class names file is empty"):
        _load_class_names_from_txt(txt_file)


def test_build_parser() -> None:
    """Test _build_parser() creates argument parser."""
    parser = _build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "d20"


def test_get_arg_value_exists() -> None:
    """Test _get_arg_value() with existing attribute."""
    args = argparse.Namespace(test_attr="value")
    result = _get_arg_value(args, "test_attr")
    assert result == "value"


def test_get_arg_value_not_exists() -> None:
    """Test _get_arg_value() with non-existent attribute."""
    args = argparse.Namespace()
    result = _get_arg_value(args, "test_attr")
    assert result is None


def test_get_arg_value_with_default() -> None:
    """Test _get_arg_value() with default value."""
    args = argparse.Namespace()
    result = _get_arg_value(args, "test_attr", "default")
    assert result == "default"


def test_get_arg_value_empty_string() -> None:
    """Test _get_arg_value() returns None for empty string."""
    args = argparse.Namespace(test_attr="")
    result = _get_arg_value(args, "test_attr")
    assert result is None


def test_update_yolo_params(tmp_path: Path) -> None:
    """Test _update_detected_params_with_cli() updates YOLO parameters."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
        annotations_dir=None,
        yaml_path=None,
        dataset_root=None,
    )
    args = argparse.Namespace(images_dir="imgs", labels_dir="lbls", annotations_dir="anns")

    updated = _update_detected_params_with_cli(params, args)
    assert isinstance(updated, YoloDetectedParams)
    assert updated.images_dir == "imgs"
    assert updated.labels_dir == "lbls"
    assert updated.annotations_dir == "anns"
    assert updated.class_names == ["cat"]  # Unchanged
    assert updated.splits == ["train"]  # Unchanged


def test_update_yolo_params_with_class_names(tmp_path: Path) -> None:
    """Test _update_detected_params_with_cli() with class names override."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    class_file = tmp_path / "classes.txt"
    class_file.write_text("dog\nbird")
    args = argparse.Namespace(class_names_file=str(class_file))

    updated = _update_detected_params_with_cli(params, args)
    assert isinstance(updated, YoloDetectedParams)
    assert updated.class_names == ["dog", "bird"]


def test_update_coco_params(tmp_path: Path) -> None:
    """Test _update_detected_params_with_cli() updates COCO parameters."""
    params = CocoDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )
    args = argparse.Namespace(images_dir="imgs", labels_dir="lbls", annotations_dir="anns")

    updated = _update_detected_params_with_cli(params, args)
    assert isinstance(updated, CocoDetectedParams)
    assert updated.images_dir == "imgs"
    assert updated.labels_dir == "lbls"
    assert updated.annotations_dir == "anns"


def test_update_coco_params_with_images_path(tmp_path: Path) -> None:
    """Test _update_detected_params_with_cli() with images_path for single JSON file."""
    json_file = tmp_path / "annotations.json"
    json_file.write_text("{}")

    params = CocoDetectedParams(
        input_path=json_file,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    args = argparse.Namespace(images_path=str(images_dir))

    updated = _update_detected_params_with_cli(params, args)
    assert isinstance(updated, CocoDetectedParams)
    assert updated.split_files is not None
    assert "annotations" in updated.split_files


def test_update_detected_params_with_cli_yolo(tmp_path: Path) -> None:
    """Test _update_detected_params_with_cli() with YOLO params."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )

    class_names_file = tmp_path / "classes.txt"
    class_names_file.write_text("cat\ndog")
    args = argparse.Namespace(
        class_names_file=str(class_names_file),
        splits="train,val",
        images_dir="imgs",
    )

    updated = _update_detected_params_with_cli(params, args)
    assert isinstance(updated, YoloDetectedParams)
    assert updated.class_names == ["cat", "dog"]
    assert updated.splits == ["train", "val"]
    assert updated.images_dir == "imgs"


def test_build_write_params_yolo(tmp_path: Path) -> None:
    """Test _build_write_params() for YOLO format."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat", "dog"],
        splits=["train"],
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace(splits="train,val", images_dir="imgs")

    write_params = _build_write_params("yolo", params, args)
    assert isinstance(write_params, YoloWriteDetectedParams)
    assert write_params.class_names == ["cat", "dog"]
    assert write_params.splits == ["train", "val"]
    assert write_params.images_dir == "imgs"


def test_build_write_params_coco(tmp_path: Path) -> None:
    """Test _build_write_params() for COCO format."""
    params = CocoDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["train"],
        images_dir="images",
        labels_dir=None,
        annotations_dir="annotations",
        split_files=None,
    )
    args = argparse.Namespace()

    write_params = _build_write_params("coco", params, args)
    assert isinstance(write_params, CocoWriteDetectedParams)
    assert write_params.class_names == ["cat"]


def test_build_write_params_missing_class_names(tmp_path: Path) -> None:
    """Test _build_write_params() raises error when class names are missing."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace()

    with pytest.raises(ValueError, match="Class names are required"):
        _build_write_params("yolo", params, args)


def test_build_write_params_unknown_format(tmp_path: Path) -> None:
    """Test _build_write_params() raises error for unknown format."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace()

    with pytest.raises(ValueError, match="Unknown format"):
        _build_write_params("unknown", params, args)


def test_ensure_class_names_with_names(tmp_path: Path) -> None:
    """Test _ensure_class_names() when class names already exist."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace()

    result = _ensure_class_names(params, args)
    assert result.class_names == ["cat"]


def test_ensure_class_names_without_names_no_file(tmp_path: Path) -> None:
    """Test _ensure_class_names() raises error when no class names and no file."""
    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace()

    with pytest.raises(ValueError, match="Class names must be provided"):
        _ensure_class_names(params, args)


def test_ensure_class_names_loads_from_file(tmp_path: Path) -> None:
    """Test _ensure_class_names() loads from file when needed."""
    txt_file = tmp_path / "classes.txt"
    txt_file.write_text("cat\ndog\n")

    params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=None,
        splits=None,
        images_dir="images",
        labels_dir="labels",
    )
    args = argparse.Namespace(class_names_file=str(txt_file))

    result = _ensure_class_names(params, args)
    assert result.class_names == ["cat", "dog"]


@patch("d20.cli.get_converter")
@patch("d20.cli.convert_dataset")
def test_handle_convert_command(mock_convert: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test _handle_convert_command() function."""
    parser = _build_parser()

    # Mock converter
    mock_converter = MagicMock()
    mock_params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )
    mock_converter.autodetect.return_value = mock_params
    mock_get_converter.return_value = mock_converter

    args = argparse.Namespace(
        source_format="yolo",
        target_format="coco",
        input=str(tmp_path),
        output=str(tmp_path / "output"),
        class_names_file=None,
        splits=None,
        images_dir=None,
        labels_dir=None,
        annotations_dir=None,
    )

    _handle_convert_command(args, parser)
    mock_convert.assert_called_once()


@patch("d20.cli.get_converter")
@patch("d20.cli.export_fiftyone")
def test_handle_export_command(mock_export: MagicMock, mock_get_converter: MagicMock, tmp_path: Path) -> None:
    """Test _handle_export_command() function."""
    # Mock converter
    mock_converter = MagicMock()
    mock_params = YoloDetectedParams(
        input_path=tmp_path,
        class_names=["cat"],
        splits=["data"],
        images_dir="images",
        labels_dir="labels",
    )
    mock_converter.autodetect.return_value = mock_params
    mock_get_converter.return_value = mock_converter

    args = argparse.Namespace(
        format="yolo",
        input=str(tmp_path),
        class_names_file=None,
        splits=None,
        images_dir=None,
        images_path=None,
        labels_dir=None,
        annotations_dir=None,
        split=None,
    )

    _handle_export_command(args)
    mock_export.assert_called_once()


def test_main_convert_command(tmp_path: Path) -> None:
    """Test main() with convert command."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    argv = [
        "convert",
        "yolo",
        "coco",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
    ]

    with patch("d20.cli._handle_convert_command") as mock_handle, patch("d20.cli._build_parser") as mock_parser:
        mock_args = argparse.Namespace(command="convert", source_format="yolo", target_format="coco")
        mock_parser.return_value.parse_args.return_value = mock_args
        main(argv)
        mock_handle.assert_called_once()


def test_main_backward_compatibility(tmp_path: Path) -> None:
    """Test main() with backward compatibility (old format)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    argv = [
        "yolo",
        "coco",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
    ]

    with patch("d20.cli._handle_convert_command") as mock_handle, patch("d20.cli._build_parser") as mock_parser:
        mock_args = argparse.Namespace(
            command="convert",
            source_format="yolo",
            target_format="coco",
            input=str(input_dir),
            output=str(output_dir),
        )
        mock_parser.return_value.parse_args.return_value = mock_args
        main(argv)
        # Should insert "convert" command and call handle
        assert mock_parser.return_value.parse_args.call_args[0][0][0] == "convert"
        mock_handle.assert_called_once()


def test_main_export_command(tmp_path: Path) -> None:
    """Test main() with export command."""
    input_dir = tmp_path / "input"
    argv = [
        "export",
        "yolo",
        "--input",
        str(input_dir),
    ]

    with patch("d20.cli._handle_export_command") as mock_handle, patch("d20.cli._build_parser") as mock_parser:
        mock_args = argparse.Namespace(command="export", format="yolo")
        mock_parser.return_value.parse_args.return_value = mock_args
        main(argv)
        mock_handle.assert_called_once()


def test_main_no_command() -> None:
    """Test main() with no command raises error."""
    argv: list[str] = []

    with patch("d20.cli._build_parser") as mock_parser:
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        mock_args = argparse.Namespace(command=None)
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser_instance.error = MagicMock(side_effect=SystemExit)

        with pytest.raises(SystemExit):
            main(argv)
        mock_parser_instance.error.assert_called_once()
