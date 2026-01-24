"""Main entry point for d20."""

from pathlib import Path

from loguru import logger

from tests.utils.downloader import DownloadConfig, download_datasets


def main() -> None:
    """Main function."""
    logger.info("Starting d20 dataset downloader")

    config = DownloadConfig(
        output_dir=Path("datasets"),
        datasets=["coco8", "coco12-formats"],
    )

    download_datasets(config)


if __name__ == "__main__":
    main()
