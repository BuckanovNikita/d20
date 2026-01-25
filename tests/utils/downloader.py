"""Tool for downloading COCO datasets from Ultralytics."""

import shutil
import zipfile
from pathlib import Path

import requests
from loguru import logger
from pydantic import BaseModel, Field


class DownloadConfig(BaseModel):
    """Configuration for dataset download."""

    output_dir: Path = Field(default=Path("datasets"), description="Output directory for datasets")
    datasets: list[str] = Field(
        default=["coco8", "coco12-formats"],
        description="List of datasets to download",
    )
    base_url: str = Field(
        default="https://github.com/ultralytics/assets/releases/download/v0.0.0",
        description="Base URL for Ultralytics assets",
    )


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL to output path.

    Args:
        url: URL to download from
        output_path: Path where file will be saved

    """
    logger.info(f"Downloading {url} to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    logger.debug(f"Progress: {percent:.1f}%")

    logger.info(f"Downloaded {output_path} ({downloaded / 1024 / 1024:.2f} MB)")


def _get_root_dirs(namelist: list[str]) -> set[str]:
    """Extract root directory names from zip file namelist."""
    root_dirs = {name.split("/")[0] for name in namelist if "/" in name}
    root_dirs.update({name.split("\\")[0] for name in namelist if "\\" in name})
    return {d for d in root_dirs if d}  # Remove empty strings


def _move_yaml_files(nested_dir: Path, extract_to: Path) -> None:
    """Move YAML files from nested directory to extract_to."""
    for yaml_file in nested_dir.glob("*.yaml"):
        yaml_dest = extract_to / yaml_file.name
        if yaml_dest.exists():
            yaml_dest.unlink()
        yaml_file.rename(yaml_dest)
        logger.info(f"Moved {yaml_file.name} to {extract_to}")


def _move_other_contents(nested_dir: Path, dataset_dir: Path) -> None:
    """Move all non-YAML contents from nested directory to dataset directory."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(exist_ok=True)

    for item in nested_dir.iterdir():
        if item.is_file() and item.suffix == ".yaml":
            continue  # Already moved
        dest = dataset_dir / item.name
        if item.is_dir():
            shutil.move(str(item), str(dest))
        else:
            item.rename(dest)


def _extract_nested_structure(
    zip_ref: zipfile.ZipFile,
    extract_to: Path,
    root_dirs: set[str],
    dataset_name_clean: str,
) -> None:
    """Extract zip with nested structure handling."""
    logger.info("Detected nested structure, extracting contents one level up")
    temp_extract = extract_to / "_temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)
    zip_ref.extractall(temp_extract)

    nested_dir = temp_extract / next(iter(root_dirs))
    if nested_dir.exists():
        _move_yaml_files(nested_dir, extract_to)
        dataset_dir = extract_to / dataset_name_clean
        _move_other_contents(nested_dir, dataset_dir)

    if temp_extract.exists():
        shutil.rmtree(temp_extract)


def extract_zip(zip_path: Path, extract_to: Path, dataset_name: str) -> None:
    """Extract a zip file to a directory, handling nested structure.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to (output_dir)
        dataset_name: Name of the dataset (to detect nested structure)

    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        namelist = zip_ref.namelist()
        if not namelist:
            logger.warning(f"Zip file {zip_path} is empty")
            return

        root_dirs = _get_root_dirs(namelist)
        dataset_name_clean = dataset_name.lower().replace("_", "-")

        if len(root_dirs) == 1 and dataset_name_clean in root_dirs:
            _extract_nested_structure(zip_ref, extract_to, root_dirs, dataset_name_clean)
        else:
            zip_ref.extractall(extract_to)

    logger.info(f"Extracted to {extract_to}")


def download_dataset(dataset_name: str, config: DownloadConfig) -> None:
    """Download a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'coco8', 'coco12-formats')
        config: Download configuration

    """
    dataset_name_clean = dataset_name.lower().replace("_", "-")
    url = f"{config.base_url}/{dataset_name_clean}.zip"
    zip_path = config.output_dir / f"{dataset_name_clean}.zip"
    dataset_dir = config.output_dir / dataset_name_clean
    yaml_path = config.output_dir / f"{dataset_name_clean}.yaml"

    # Skip if already extracted (check for both yaml and dataset directory)
    if yaml_path.exists() and dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info(f"Dataset {dataset_name_clean} already exists, skipping")
        return

    # Download zip file
    if not zip_path.exists():
        try:
            download_file(url, zip_path)
        except requests.HTTPError as e:
            logger.error(f"Failed to download {dataset_name_clean}: {e}")
            if e.response.status_code == 404:
                logger.warning(f"Dataset {dataset_name_clean} not found at {url}")
            raise

    # Extract zip file
    if zip_path.exists():
        extract_zip(zip_path, config.output_dir, dataset_name_clean)
        # Optionally remove zip file after extraction
        zip_path.unlink()
        logger.info(f"Removed zip file {zip_path}")

    # For coco8, download yaml file if it doesn't exist
    if dataset_name_clean == "coco8":
        yaml_path = config.output_dir / "coco8.yaml"
        if not yaml_path.exists():
            logger.info("Downloading coco8.yaml configuration file")
            yaml_url = (
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml"
            )
            try:
                download_file(yaml_url, yaml_path)
            except (requests.HTTPError, requests.RequestException, OSError) as e:
                logger.warning(f"Failed to download coco8.yaml: {e}")


def download_datasets(config: DownloadConfig | None = None) -> None:
    """Download COCO datasets from Ultralytics.

    Args:
        config: Download configuration. If None, uses default config.

    """
    if config is None:
        config = DownloadConfig()

    logger.info(f"Starting download of datasets: {config.datasets}")
    logger.info(f"Output directory: {config.output_dir}")

    try:
        for dataset in config.datasets:
            download_dataset(dataset, config)
    except (requests.HTTPError, requests.RequestException, OSError, ValueError) as e:
        logger.error(f"Error downloading datasets: {e}")
        raise

    logger.info("Download completed")


if __name__ == "__main__":
    download_datasets()
