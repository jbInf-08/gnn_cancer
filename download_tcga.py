import argparse
from pathlib import Path
from data_sources.clinical_repos import TCGADownloader

def main():
    parser = argparse.ArgumentParser(description="Download TCGA data")
    parser.add_argument("--dest", type=str, default="data/raw/tcga",
                      help="Destination directory for downloaded files")
    args = parser.parse_args()
    
    # Create downloader
    downloader = TCGADownloader()
    
    # Download data
    dest = Path(args.dest)
    downloader.download("tcga", dest)

if __name__ == "__main__":
    main() 