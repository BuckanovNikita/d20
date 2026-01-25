"""Command-line interface for d20 dataset conversion tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

from d20.config import ConversionConfig
from d20.convert import convert_dataset
from d20.export_cli import add_export_parser, handle_export_command

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


def _extract_class_names_from_yolo_yaml(yaml_path: Path) -> list[str] | None:
    """Extract class names from YOLO YAML file if it exists and contains names."""
    if not yaml_path.exists() or yaml_path.suffix.lower() not in (".yaml", ".yml"):
        return None

    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
        names = data.get("names")
        if not names:
            return None

        if isinstance(names, list):
            return names

        if isinstance(names, dict):

            def _key_as_int(key: str | int) -> int:
                return int(key)

            return [names[key] for key in sorted(names.keys(), key=_key_as_int)]
    except (yaml.YAMLError, KeyError, ValueError, TypeError):
        return None
    else:
        return None


def _extract_categories_from_json(json_path: Path) -> tuple[set[str], dict[str, int]]:
    """Extract category names and mapping from a COCO JSON file.

    Returns:
        Tuple of (set of category names, dict mapping name to id)

    """
    all_names: set[str] = set()
    category_map: dict[str, int] = {}

    try:
        data = json.loads(json_path.read_text())
        categories = data.get("categories", [])
        for category in categories:
            name = category.get("name")
            cat_id = category.get("id")
            if name:
                all_names.add(name)
                if cat_id is not None and name not in category_map:
                    category_map[name] = int(cat_id)
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
        logger.debug("Failed to extract categories from {}: {}", json_path, e)

    return all_names, category_map


def _extract_class_names_from_coco_dataset(
    input_path: Path, annotations_dir: str, splits: list[str]
) -> list[str] | None:
    """Extract class names from COCO dataset JSON files."""
    input_dir = Path(input_path)
    all_category_names: set[str] = set()
    category_map: dict[str, int] = {}

    for split_name in splits:
        json_path = input_dir / annotations_dir / f"{split_name}.json"
        if not json_path.exists():
            continue
        names, mapping = _extract_categories_from_json(json_path)
        all_category_names.update(names)
        category_map.update(mapping)

    if not all_category_names:
        return None

    # Sort by category id if available, otherwise alphabetically
    if category_map:
        sorted_names = sorted(all_category_names, key=lambda n: category_map.get(n, 999))
    else:
        sorted_names = sorted(all_category_names)

    return sorted_names if sorted_names else None


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
    convert_parser.add_argument("--splits", help="Comma-separated split names (default: data)")
    convert_parser.add_argument("--images-dir", default="images", help="Images directory name (default: images)")
    convert_parser.add_argument("--labels-dir", default="labels", help="Labels directory name (default: labels)")
    convert_parser.add_argument(
        "--annotations-dir", default="annotations", help="Annotations directory name (default: annotations)"
    )

    # Export command
    add_export_parser(subparsers)

    # Backward compatibility: if first arg is a format, treat as convert command
    # This allows: d20 yolo coco --input ... (old format)
    parser.set_defaults(command=None)

    return parser


def _extract_yolo_class_names_from_path(yaml_path: Path) -> list[str] | None:
    """Extract class names from YOLO YAML file or directory."""
    if yaml_path.suffix.lower() in (".yaml", ".yml") and yaml_path.exists():
        return _extract_class_names_from_yolo_yaml(yaml_path)

    if yaml_path.is_dir():
        for pattern in ("*.yaml", "*.yml"):
            for yaml_file in yaml_path.glob(pattern):
                class_names = _extract_class_names_from_yolo_yaml(yaml_file)
                if class_names:
                    return class_names
    return None


def _resolve_class_names(
    source_format: str,
    input_path: Path,
    class_names_file: str | None,
    annotations_dir: str,
    splits: list[str],
) -> list[str]:
    """Resolve class names from various sources based on format."""
    # Priority 1: --class-names-file argument
    if class_names_file:
        return _load_class_names_from_txt(Path(class_names_file))

    # Priority 2: Format-specific extraction
    if source_format == "yolo":
        class_names = _extract_yolo_class_names_from_path(Path(input_path))
        if class_names:
            return class_names
    elif source_format == "coco":
        class_names = _extract_class_names_from_coco_dataset(input_path, annotations_dir, splits)
        if class_names:
            return class_names

    # If we get here, class names couldn't be determined
    msg = (
        "Class names must be provided. Use --class-names-file "
        "or ensure the dataset contains class information (YOLO YAML or COCO JSON)"
    )
    raise ValueError(msg)


def _build_config(args: argparse.Namespace) -> ConversionConfig:
    """Build ConversionConfig from CLI arguments."""
    input_path = Path(args.input)
    splits = _parse_list(args.splits) if args.splits else ["data"]

    class_names = _resolve_class_names(
        source_format=args.source_format,
        input_path=input_path,
        class_names_file=getattr(args, "class_names_file", None),
        annotations_dir=args.annotations_dir,
        splits=splits,
    )

    config_data = {
        "class_names": class_names,
        "images_dir": args.images_dir,
        "labels_dir": args.labels_dir,
        "annotations_dir": args.annotations_dir,
    }

    if args.splits:
        config_data["splits"] = splits

    return ConversionConfig.model_validate(config_data)


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
        # Handle backward compatibility case
        if not hasattr(args, "source_format"):
            parser.error("Missing source and target formats")

        config = _build_config(args)

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
    elif args.command == "export":
        handle_export_command(args)
    else:
        parser.error("Please specify a command: convert or export")
