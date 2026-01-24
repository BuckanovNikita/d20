"""Tool for downloading COCO datasets from Ultralytics."""

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


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file to a directory.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    import zipfile

    logger.info(f"Extracting {zip_path} to {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
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
    extract_dir = config.output_dir / dataset_name_clean

    # Skip if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info(f"Dataset {dataset_name_clean} already exists at {extract_dir}, skipping")
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
        extract_zip(zip_path, extract_dir)
        # Optionally remove zip file after extraction
        zip_path.unlink()
        logger.info(f"Removed zip file {zip_path}")


def download_datasets(config: DownloadConfig | None = None) -> None:
    """Download COCO datasets from Ultralytics.

    Args:
        config: Download configuration. If None, uses default config.
    """
    if config is None:
        config = DownloadConfig()

    logger.info(f"Starting download of datasets: {config.datasets}")
    logger.info(f"Output directory: {config.output_dir}")

    for dataset in config.datasets:
        try:
            download_dataset(dataset, config)
        except Exception as e:
            logger.error(f"Error downloading {dataset}: {e}")
            continue

    logger.info("Download completed")


if __name__ == "__main__":
    download_datasets()
