import argparse
from pathlib import Path
from data_sources.public_repos import NCBIDownloader

def main():
    parser = argparse.ArgumentParser(description="Download NCBI datasets")
    parser.add_argument("--dataset", type=str, default="gene",
                      help="Dataset to download (e.g., gene, pubmed, sra)")
    parser.add_argument("--dest", type=str, default="data/raw/ncbi",
                      help="Destination directory for downloaded files")
    args = parser.parse_args()
    
    downloader = NCBIDownloader()
    dest = Path(args.dest)
    downloader.download(args.dataset, dest)

if __name__ == "__main__":
    main() 