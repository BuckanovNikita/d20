# Dataset Format Specifications

This document provides comprehensive specifications for the object detection dataset formats supported by d20: Ultralytics YOLO and COCO.

---

## Table of Contents

1. [Ultralytics YOLO Format](#ultralytics-yolo-format)
2. [COCO Format](#coco-format)

---

## Ultralytics YOLO Format

### Overview

The Ultralytics YOLO format is the official dataset configuration format for training YOLO models. It uses a directory-based structure with YAML configuration files and text-based label files.

### Directory Structure

```
my-dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       ├── img003.jpg
│       └── img004.jpg
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   └── val/
│       ├── img003.txt
│       └── img004.txt
└── data.yaml
```

### YAML Configuration File (`data.yaml`)

The `data.yaml` file defines the dataset configuration. It must contain:

**Required fields:**
- `path`: Dataset root directory (relative or absolute path)
- `train`: Path to training images (relative to root)
- `val`: Path to validation images (relative to root)
- `names`: Dictionary or list of class names indexed from 0

**Optional fields:**
- `test`: Path to test images (relative to root)

**Example:**
```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: car
  2: dog
```

**Alternative names format (list):**
```yaml
names:
  - person
  - car
  - dog
```

### Label Format

Labels are stored in `*.txt` files corresponding to each image, with one row per object.

**Format:**
```
class_id x_center y_center width height
```

**Requirements:**
- **Coordinates must be normalized** (values from 0.0 to 1.0)
- Divide pixel coordinates by image width/height to normalize
- **Class numbers are zero-indexed** (starting from 0)
- If an image has no objects, the label file may be omitted or empty

**Example label file (`img001.txt`):**
```
0 0.5 0.5 0.2 0.3
1 0.1 0.1 0.15 0.2
```

**Coordinate system:**
- `x_center`: Normalized x-coordinate of bounding box center (0.0 to 1.0)
- `y_center`: Normalized y-coordinate of bounding box center (0.0 to 1.0)
- `width`: Normalized width of bounding box (0.0 to 1.0)
- `height`: Normalized height of bounding box (0.0 to 1.0)

**Conversion from pixel coordinates:**
```
x_center = (x_min + x_max) / 2 / image_width
y_center = (y_min + y_max) / 2 / image_height
width = (x_max - x_min) / image_width
height = (y_max - y_min) / image_height
```

### Supported Image Formats

Ultralytics YOLO supports the following image formats:
- **JPEG**: `.jpg`, `.jpeg` (most common, recommended)
- **PNG**: `.png` (supports transparency)
- **WebP**: `.webp` (modern, good compression)
- **BMP**: `.bmp` (uncompressed)
- **GIF**: `.gif` (first frame extracted)
- **TIFF**: `.tiff`, `.tif` (high quality)
- **HEIC**: `.heic` (iPhone photos)
- **AVIF**: `.avif` (next-gen format)
- **JP2**: `.jp2` (JPEG 2000)
- **DNG**: `.dng` (raw camera)

### File Size Limits

- **Images**: 50 MB each
- **ZIP files**: 50 GB

### Validation Rules

1. All coordinates must be in the range [0.0, 1.0]
2. Class IDs must be non-negative integers within the range of defined classes
3. Calculated bounding boxes must be within image bounds
4. Width and height must be positive values
5. Label file names must match image file names (with `.txt` extension)

### Extended Format Support

Ultralytics YOLO format also supports additional task types beyond object detection:

**Segmentation:**
```
class_id x1 y1 x2 y2 x3 y3 ...
```

**Pose estimation:**
```
class_id cx cy w h kx1 ky1 v1 kx2 ky2 v2 ...
```
- Visibility flags: `0` = not labeled, `1` = labeled but occluded, `2` = labeled and visible

**Oriented Bounding Boxes (OBB):**
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

**Classification:**
Uses directory structure: `train/cats/`, `train/dogs/`, etc.

---

## COCO Format

### Overview

COCO (Common Objects in Context) is a widely-used JSON-based format for object detection, instance segmentation, and keypoint detection tasks. The format was introduced by Microsoft in 2015 and has become a standard benchmark format.

### File Structure

COCO datasets use a single JSON file (or multiple JSON files for different splits) containing all annotations and metadata.

### JSON Schema

A COCO JSON file contains five main sections:

#### 1. `info` (Optional)

General metadata about the dataset:

```json
{
  "info": {
    "year": 2024,
    "version": "1.0",
    "description": "Dataset description",
    "contributor": "Contributor name",
    "url": "https://example.com",
    "date_created": "2024-01-01T00:00:00Z"
  }
}
```

#### 2. `licenses` (Optional)

License information for images:

```json
{
  "licenses": [
    {
      "id": 1,
      "name": "CC BY 4.0",
      "url": "https://creativecommons.org/licenses/by/4.0/"
    }
  ]
}
```

#### 3. `images` (Required)

Array of image objects with metadata:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000000000009.jpg",
      "width": 640,
      "height": 480,
      "date_captured": "2024-01-01T00:00:00Z",
      "license": 1,
      "flickr_url": "https://example.com/image.jpg",
      "coco_url": "https://example.com/image.jpg"
    }
  ]
}
```

**Required fields:**
- `id`: Unique integer identifier for the image
- `file_name`: Image filename (including extension)
- `width`: Image width in pixels (must be > 0)
- `height`: Image height in pixels (must be > 0)

**Optional fields:**
- `date_captured`: ISO 8601 timestamp
- `license`: License ID (references `licenses` array)
- `flickr_url`: URL to image on Flickr
- `coco_url`: URL to image on COCO website

#### 4. `annotations` (Required)

Array of annotation objects:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 150, 180],
      "area": 27000.0,
      "segmentation": [[100, 200, 250, 200, 250, 380, 100, 380]],
      "iscrowd": 0
    }
  ]
}
```

**Required fields:**
- `id`: Unique integer identifier for the annotation
- `image_id`: Integer ID linking to an image in the `images` array
- `category_id`: Integer ID linking to a category in the `categories` array
- `bbox`: Bounding box as `[x, y, width, height]` in pixel coordinates
  - `x`: Left-most pixel coordinate
  - `y`: Top-most pixel coordinate
  - `width`: Width of bounding box in pixels
  - `height`: Height of bounding box in pixels
- `area`: Float value representing the area of the annotation in pixels²
- `iscrowd`: Integer (0 or 1) indicating whether the instance is a group of objects

**Optional fields:**
- `segmentation`: Polygon or RLE-encoded mask for instance segmentation
  - **Polygon format**: Array of arrays `[[x1, y1, x2, y2, ...]]` representing polygon vertices
  - **RLE format**: Run-length encoded mask (used when `iscrowd=1`)
- `keypoints`: Array for keypoint detection tasks
- `num_keypoints`: Number of keypoints for pose estimation

**Bounding box format:**
- Coordinates are in **absolute pixel values** (not normalized)
- Format is `[x, y, width, height]` (XYWH format)
- All values must be non-negative
- Bounding box must be within image bounds

**Area calculation:**
- For bounding boxes: `area = width × height`
- For segmentation: `area` should match the actual segmented area

**IsCrowd attribute:**
- `iscrowd=0`: Single object instance (uses polygon segmentation)
- `iscrowd=1`: Group of objects (uses RLE-encoded mask)
- When `iscrowd=1`, all shapes within the group convert into a single mask

#### 5. `categories` (Required)

Array of category/class definitions:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

**Required fields:**
- `id`: Unique integer identifier for the category
- `name`: Category name (string)

**Optional fields:**
- `supercategory`: Parent category grouping (string)

### Directory Structure

COCO datasets can be organized in two ways:

**Option 1: Single JSON file**
```
dataset/
├── annotations/
│   └── instances_train.json
└── images/
    └── train/
        ├── 000000000009.jpg
        └── 000000000025.jpg
```

**Option 2: Multiple JSON files (one per split)**
```
dataset/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── train/
    │   ├── 000000000009.jpg
    │   └── 000000000025.jpg
    ├── val/
    │   └── 000000000036.jpg
    └── test/
        └── 000000000042.jpg
```

### Validation Rules

1. All `image_id` values in annotations must reference valid images
2. All `category_id` values in annotations must reference valid categories
3. Bounding boxes must have 4 elements: `[x, y, width, height]`
4. All bbox values must be non-negative
5. Bounding boxes must be within image bounds
6. `area` must be positive
7. `iscrowd` must be 0 or 1
8. Category IDs must be unique
9. Image IDs must be unique

### Task-Specific Extensions

**Instance Segmentation:**
- Requires `segmentation` field with polygon or RLE data
- `area` should match segmented area, not bounding box area

**Keypoint Detection:**
- Requires `keypoints` array: `[x1, y1, v1, x2, y2, v2, ...]`
- Visibility flags: `0` = not labeled, `1` = labeled but occluded, `2` = labeled and visible
- Requires `num_keypoints` field

**Panoptic Segmentation:**
- Uses separate `panoptic_annotations` section
- Combines instance segmentation with stuff segmentation

### Official References

- [COCO Dataset Website](https://cocodataset.org/)
- [COCO API GitHub Repository](https://github.com/cocodataset/cocoapi)
- [COCO Format Documentation](https://cocodataset.org/#format-data)

---

## Format Comparison

| Feature | YOLO | COCO |
|---------|------|------|
| **File Format** | Text files + YAML | JSON |
| **Coordinate System** | Normalized (0-1) | Absolute pixels |
| **Bbox Format** | XYWH (normalized) | XYWH (pixels) |
| **Class Definition** | YAML `names` field | JSON `categories` array |
| **Split Organization** | Directory structure | JSON file(s) |
| **Segmentation Support** | Polygon format | Polygon + RLE |
| **Metadata** | Minimal | Rich (licenses, info) |
| **Multi-file** | Yes (one label per image) | No (single/multiple JSON) |

---

## Notes

- All formats support object detection as the primary task
- COCO format has the most extensive support for additional tasks (segmentation, keypoints)
- YOLO format is optimized for training YOLO models and uses normalized coordinates
- When converting between formats, coordinate transformations and class mappings must be handled carefully
