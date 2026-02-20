# Script to download a uci dataset
import argparse
import zipfile
from pathlib import Path

import requests  # type: ignore[import-untyped]


def download_dataset(url: str, dataset_name: str, base_dir: str = "./datasets"):
    """
    Downloads a dataset from a given URL and saves it under base_dir/dataset_name.
    If the file is a .zip archive, it will be automatically extracted.
    """
    dataset_dir = Path(base_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    download_path = dataset_dir / filename

    print(f"Downloading {dataset_name} from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {download_path}")

    if zipfile.is_zipfile(download_path):
        print("Extracting archive...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"Extracted to {dataset_dir}")
        download_path.unlink()

    print(f"Dataset ready in: {dataset_dir}")
    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from a URL.")
    parser.add_argument("url", type=str, help="Direct download URL for the dataset.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name (default: derived from filename).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./experiments/uci_datasets",
        help="Base directory where datasets are stored.",
    )
    args = parser.parse_args()

    dataset_name = args.name or Path(args.url).stem
    download_dataset(args.url, dataset_name, base_dir=args.base_dir)
