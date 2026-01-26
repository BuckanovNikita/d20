"""Tests for type definitions and Dataset utility methods."""

from __future__ import annotations

from pathlib import Path

from d20.types import Annotation, Dataset, ImageInfo, Split


def test_dataset_get_unique_image_sizes() -> None:
    """Test Dataset.get_unique_image_sizes() method."""
    images1 = [
        ImageInfo(image_id="1", file_name="img1.jpg", width=100, height=200, path=Path("img1.jpg")),
        ImageInfo(image_id="2", file_name="img2.jpg", width=100, height=200, path=Path("img2.jpg")),
    ]
    images2 = [
        ImageInfo(image_id="3", file_name="img3.jpg", width=300, height=400, path=Path("img3.jpg")),
    ]
    split1 = Split(name="train", images=images1, annotations=[])
    split2 = Split(name="val", images=images2, annotations=[])

    dataset = Dataset(splits={"train": split1, "val": split2}, class_names=["class1"])

    unique_sizes = dataset.get_unique_image_sizes()
    assert unique_sizes == {(100, 200), (300, 400)}


def test_dataset_get_unique_image_sizes_empty() -> None:
    """Test Dataset.get_unique_image_sizes() with empty dataset."""
    dataset = Dataset(splits={}, class_names=[])
    unique_sizes = dataset.get_unique_image_sizes()
    assert unique_sizes == set()


def test_dataset_get_available_splits() -> None:
    """Test Dataset.get_available_splits() method."""
    split1 = Split(name="train", images=[], annotations=[])
    split2 = Split(name="val", images=[], annotations=[])
    split3 = Split(name="test", images=[], annotations=[])

    dataset = Dataset(splits={"val": split2, "train": split1, "test": split3}, class_names=[])
    splits = dataset.get_available_splits()
    assert splits == ["test", "train", "val"]


def test_dataset_get_available_splits_empty() -> None:
    """Test Dataset.get_available_splits() with empty dataset."""
    dataset = Dataset(splits={}, class_names=[])
    splits = dataset.get_available_splits()
    assert splits == []


def test_dataset_get_available_classes() -> None:
    """Test Dataset.get_available_classes() method."""
    annotations1 = [
        Annotation(image_id="1", category_id=0, bbox=(10, 20, 30, 40)),
        Annotation(image_id="1", category_id=1, bbox=(50, 60, 70, 80)),
        Annotation(image_id="2", category_id=0, bbox=(15, 25, 35, 45)),
    ]
    annotations2 = [
        Annotation(image_id="3", category_id=2, bbox=(100, 110, 120, 130)),
    ]
    split1 = Split(name="train", images=[], annotations=annotations1)
    split2 = Split(name="val", images=[], annotations=annotations2)

    dataset = Dataset(splits={"train": split1, "val": split2}, class_names=["cat", "dog", "bird"])

    classes = dataset.get_available_classes()
    assert classes == ["cat", "dog", "bird"]


def test_dataset_get_available_classes_sorted_by_id() -> None:
    """Test Dataset.get_available_classes() returns classes sorted by category ID."""
    annotations = [
        Annotation(image_id="1", category_id=2, bbox=(10, 20, 30, 40)),
        Annotation(image_id="2", category_id=0, bbox=(50, 60, 70, 80)),
        Annotation(image_id="3", category_id=1, bbox=(100, 110, 120, 130)),
    ]
    split = Split(name="train", images=[], annotations=annotations)
    dataset = Dataset(splits={"train": split}, class_names=["cat", "dog", "bird"])

    classes = dataset.get_available_classes()
    assert classes == ["cat", "dog", "bird"]


def test_dataset_get_available_classes_skips_invalid_ids() -> None:
    """Test Dataset.get_available_classes() skips invalid category IDs."""
    annotations = [
        Annotation(image_id="1", category_id=0, bbox=(10, 20, 30, 40)),
        Annotation(image_id="2", category_id=5, bbox=(50, 60, 70, 80)),  # Invalid ID
        Annotation(image_id="3", category_id=-1, bbox=(100, 110, 120, 130)),  # Invalid ID
    ]
    split = Split(name="train", images=[], annotations=annotations)
    dataset = Dataset(splits={"train": split}, class_names=["cat", "dog"])

    classes = dataset.get_available_classes()
    assert classes == ["cat"]


def test_dataset_get_available_classes_empty() -> None:
    """Test Dataset.get_available_classes() with no annotations."""
    dataset = Dataset(splits={}, class_names=["cat", "dog"])
    classes = dataset.get_available_classes()
    assert classes == []


def test_dataset_post_init_computes_unique_sizes() -> None:
    """Test Dataset.__post_init__() computes unique image sizes."""
    images = [
        ImageInfo(image_id="1", file_name="img1.jpg", width=100, height=200, path=Path("img1.jpg")),
        ImageInfo(image_id="2", file_name="img2.jpg", width=100, height=200, path=Path("img2.jpg")),
    ]
    split = Split(name="train", images=images, annotations=[])
    dataset = Dataset(splits={"train": split}, class_names=["class1"])

    # Verify that _unique_image_sizes was computed
    assert dataset._unique_image_sizes == {(100, 200)}


def test_dataset_post_init_with_existing_sizes() -> None:
    """Test Dataset.__post_init__() with pre-computed sizes."""
    images = [
        ImageInfo(image_id="1", file_name="img1.jpg", width=100, height=200, path=Path("img1.jpg")),
    ]
    split = Split(name="train", images=images, annotations=[])
    dataset = Dataset(
        splits={"train": split},
        class_names=["class1"],
        _unique_image_sizes={(50, 50)},  # Pre-computed
    )

    # Should use pre-computed value
    assert dataset._unique_image_sizes == {(50, 50)}
    assert dataset.get_unique_image_sizes() == {(50, 50)}
