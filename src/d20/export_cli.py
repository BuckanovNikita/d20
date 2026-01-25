"""CLI command for exporting datasets to FiftyOne App."""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from d20.config import ConversionConfig
from d20.reporting import export_to_fiftyone

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


def _detect_coco_splits(input_path: Path, annotations_dir: str) -> list[str]:
    """Detect available splits from COCO dataset directory structure.

    Args:
        input_path: Path to dataset directory or JSON file
        annotations_dir: Annotations directory name

    Returns:
        List of detected split names

    """
    input_path_obj = Path(input_path)

    # If input is a JSON file, return its stem as the split
    if input_path_obj.is_file() and input_path_obj.suffix.lower() == ".json":
        return [input_path_obj.stem]

    # If input is a directory, look for JSON files in annotations_dir
    if input_path_obj.is_dir():
        annotations_path = input_path_obj / annotations_dir
        if annotations_path.exists():
            splits = [json_file.stem for json_file in annotations_path.glob("*.json")]
            if splits:
                return sorted(splits)

    return []


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
    input_path_obj = Path(input_path)
    all_category_names: set[str] = set()
    category_map: dict[str, int] = {}

    # Handle case where input is a JSON file directly
    if input_path_obj.is_file() and input_path_obj.suffix.lower() == ".json":
        if input_path_obj.exists():
            names, mapping = _extract_categories_from_json(input_path_obj)
            all_category_names.update(names)
            category_map.update(mapping)
    else:
        # Handle case where input is a directory
        for split_name in splits:
            json_path = input_path_obj / annotations_dir / f"{split_name}.json"
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
    dataset_format: str,
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
    if dataset_format == "yolo":
        class_names = _extract_yolo_class_names_from_path(Path(input_path))
        if class_names:
            return class_names
    elif dataset_format == "coco":
        class_names = _extract_class_names_from_coco_dataset(input_path, annotations_dir, splits)
        if class_names:
            return class_names

    # If we get here, class names couldn't be determined
    msg = (
        "Class names must be provided. Use --config, --class-names-file "
        "or ensure the dataset contains class information (YOLO YAML or COCO JSON)"
    )
    raise ValueError(msg)


def add_export_parser(subparsers: Any) -> None:  # argparse._SubParsersAction[argparse.ArgumentParser]
    """Add export subcommand to the parser.

    Args:
        subparsers: Subparsers action from main parser

    """
    export_parser = subparsers.add_parser("export", help="Export dataset to FiftyOne App for visual inspection")
    export_parser.add_argument("format", choices=FORMATS, help="Dataset format (coco, voc, yolo)")
    export_parser.add_argument("--input", required=True, help="Input dataset path")
    export_parser.add_argument("--class-names-file", help="Path to text file with class names (one per line)")
    export_parser.add_argument("--splits", help="Comma-separated split names (default: data)")
    export_parser.add_argument("--images-dir", default="images", help="Images directory name (default: images)")
    export_parser.add_argument("--images-path", help="Path to images directory (for single JSON file input)")
    export_parser.add_argument("--labels-dir", default="labels", help="Labels directory name (default: labels)")
    export_parser.add_argument(
        "--annotations-dir", default="annotations", help="Annotations directory name (default: annotations)"
    )
    export_parser.add_argument("--split", help="Specific split to export (default: all splits)")


def handle_export_command(args: Any) -> None:
    """Handle the export command.

    Args:
        args: Parsed command-line arguments

    """
    input_path = Path(args.input)

    # Auto-detect splits for COCO format if not specified
    if args.format == "coco" and not args.splits:
        detected_splits = _detect_coco_splits(input_path, args.annotations_dir)
        splits = detected_splits or ["data"]
    else:
        splits = _parse_list(args.splits) if args.splits else ["data"]

    # Handle COCO format: if input is a single JSON file, prepare it for the converter
    read_options: dict[str, Any] = {}
    actual_input_path: Path | list[Path] = input_path

    if args.format == "coco" and input_path.is_file() and input_path.suffix.lower() == ".json":
        # Single JSON file input
        split_name = input_path.stem
        if not args.splits:
            splits = [split_name]

        # If images-path is provided, use split_files with images directory as base
        if args.images_path:
            images_path = Path(args.images_path)
            # Use split_files: input_dir will be the images directory (or its parent)
            # This allows the converter to find images relative to that directory
            actual_input_path = images_path if images_path.is_dir() else images_path.parent
            read_options["split_files"] = {split_name: input_path}
        else:
            # No images-path: pass as list, converter will use JSON parent as base_dir
            actual_input_path = [input_path]
            read_options["split_names"] = splits

    # Auto-detect class names and build config (same as convert command)
    class_names = _resolve_class_names(
        dataset_format=args.format,
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
        "splits": splits,
    }

    config = ConversionConfig.model_validate(config_data)

    logger.info("Exporting {} dataset to FiftyOne App", args.format)
    export_to_fiftyone(
        dataset_format=args.format,
        input_path=actual_input_path,
        config=config,
        split=args.split,
        **read_options,
    )
