import argparse
from pathlib import Path
from data_sources.kaggle_repos import KaggleDownloader

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets")
    parser.add_argument("--dest", type=str, default="data/raw/kaggle",
                      help="Destination directory for downloaded files")
    args = parser.parse_args()
    
    # Create downloader
    downloader = KaggleDownloader()
    
    # Download data
    dest = Path(args.dest)
    downloader.download("cancer_datasets", dest)

if __name__ == "__main__":
    main() 