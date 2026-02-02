# Agent Documentation

## Project Overview

**d20** - CLI tool and library for converting object detection datasets between COCO and YOLO formats.

## Architecture

### Core Components

- **Formats** (`src/d20/formats/`): Format converters implementing `BaseFormatConverter`
  - `yolo.py` - YOLO format (Ultralytics-style)
  - `coco.py` - COCO format

- **Conversion Pipeline** (`src/d20/convert.py`):
  - `ConversionRequest` - conversion parameters
  - `convert_dataset()` - main conversion function (read → write)

- **Types** (`src/d20/types.py`):
  - `Dataset` - universal in-memory format
  - `Split`, `ImageInfo`, `Annotation` - dataset structures
  - `*DetectedParams` - format-specific autodetection results
  - `*WriteDetectedParams` - format-specific write parameters

- **CLI** (`src/d20/cli.py`):
  - `convert` - convert between formats
  - `export` - export to FiftyOne App for visualization

### Data Flow

```
Input Format → autodetect() → DetectedParams → read() → Dataset → write() → Output Format
```

## Current State

### Implemented

- ✅ Two-way conversion: COCO ↔ YOLO
- ✅ Autodetection of dataset structure (splits, directories, classes)
- ✅ CLI with `convert` and `export` commands
- ✅ FiftyOne integration for visualization
- ✅ Support for custom directory structures
- ✅ Single JSON file COCO support

### Key Design Decisions

- Universal `Dataset` format as intermediate representation
- Protocol-based type system for format-specific params
- Autodetection before conversion (can be overridden via CLI)
- Pathlib-based path handling
- Pydantic for configuration validation

### Testing

- Test fixtures in `tests/fixtures/datasets/`
- Conversion tests in `tests/test_conversions.py`
- Coverage target: 80%

## Development Guidelines

See `CLAUDE.md` for coding standards and project policies.
