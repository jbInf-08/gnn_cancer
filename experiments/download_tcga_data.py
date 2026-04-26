import requests
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_tcga_data():
    """Download TCGA breast cancer data from GDC Data Portal."""
    # Create output directory
    output_dir = Path("data/raw/tcga")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for GDC API
    base_url = "https://api.gdc.cancer.gov"
    
    # Query for BRCA expression data
    expression_query = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": "TCGA-BRCA"
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.data_type",
                        "value": "Gene Expression Quantification"
                    }
                }
            ]
        },
        "format": "JSON",
        "size": "100"
    }
    
    # Query for BRCA clinical data
    clinical_query = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": "TCGA-BRCA"
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.data_type",
                        "value": "Clinical Supplement"
                    }
                }
            ]
        },
        "format": "JSON",
        "size": "100"
    }
    
    # Query for BRCA mutation data
    mutation_query = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": "TCGA-BRCA"
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.data_type",
                        "value": "Masked Somatic Mutation"
                    }
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": "WXS"
                    }
                }
            ]
        },
        "format": "JSON",
        "size": "100"
    }
    
    def download_file(file_id, file_name, data_type):
        """Download a file from GDC API with retry logic."""
        url = f"{base_url}/data/{file_id}"
        output_path = output_dir / file_name
        
        if output_path.exists():
            logging.info(f"File {file_name} already exists, skipping...")
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(output_path, 'wb') as f, tqdm(
                    desc=f"Downloading {data_type} data",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(block_size):
                        size = f.write(data)
                        pbar.update(size)
                
                logging.info(f"Successfully downloaded {file_name}")
                return
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to download {file_name} after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed, retrying...")
    
    # Download expression data
    logging.info("Downloading expression data...")
    response = requests.post(f"{base_url}/files", json=expression_query)
    response.raise_for_status()
    files = response.json()["data"]["hits"]
    
    for file in files:
        file_id = file["file_id"]
        file_name = f"BRCA_expression_{file_id}.tsv"
        download_file(file_id, file_name, "expression")
    
    # Download clinical data
    logging.info("Downloading clinical data...")
    response = requests.post(f"{base_url}/files", json=clinical_query)
    response.raise_for_status()
    files = response.json()["data"]["hits"]
    
    for file in files:
        file_id = file["file_id"]
        file_name = f"BRCA_clinical_{file_id}.tsv"
        download_file(file_id, file_name, "clinical")
    
    # Download mutation data
    logging.info("Downloading mutation data...")
    response = requests.post(f"{base_url}/files", json=mutation_query)
    response.raise_for_status()
    files = response.json()["data"]["hits"]
    
    for file in files:
        file_id = file["file_id"]
        file_name = f"BRCA_mutation_{file_id}.maf"
        download_file(file_id, file_name, "mutation")
    
    logging.info("TCGA data download completed successfully")

if __name__ == "__main__":
    download_tcga_data() 