import requests
import gzip
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_ncbi_data():
    """Download NCBI gene data files."""
    # Create output directory
    output_dir = Path("data/raw/ncbi")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NCBI FTP URLs
    urls = {
        'gene_info': 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz',
        'gene2go': 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz',
        'gene2pubmed': 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz'
    }
    
    def download_and_extract(url, output_path):
        """Download and extract a gzipped file."""
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {output_path}")
    
    # Download each file
    for name, url in urls.items():
        output_path = output_dir / f"{name}.gz"
        download_and_extract(url, output_path)
    
    logger.info("NCBI data download completed!")

if __name__ == "__main__":
    download_ncbi_data() 