import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_uci_dataset(dataset: str, dest: Path):
    """Download UCI dataset directly from the UCI repository."""
    dest.mkdir(parents=True, exist_ok=True)
    
    if dataset == "breast_cancer_wisconsin":
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
        files = ["wdbc.data", "wdbc.names"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    for file in tqdm(files, desc=f"Downloading {dataset}"):
        url = f"{base_url}{file}"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = dest / file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {file} to {file_path}")
        else:
            logger.error(f"Failed to download {file}: {response.status_code}")

def main():
    parser = argparse.ArgumentParser(description="Download UCI datasets")
    parser.add_argument("--dataset", type=str, default="breast_cancer_wisconsin",
                      help="Dataset to download (e.g., breast_cancer_wisconsin)")
    parser.add_argument("--dest", type=str, default="data/raw/uci",
                      help="Destination directory for downloaded files")
    args = parser.parse_args()
    
    dest = Path(args.dest) / args.dataset
    download_uci_dataset(args.dataset, dest)

if __name__ == "__main__":
    main() 