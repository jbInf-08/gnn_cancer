import os
import requests
import json
import time
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_gdc_for_overlapping_samples():
    """Query GDC API to find samples that have both expression and CNV data."""
    logger.info("Querying GDC for samples with both expression and CNV data...")
    
    # GDC API endpoint
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    # Query for samples with both RNA-Seq expression and CNV data
    query = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-BRCA"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": ["Gene Expression Quantification", "Copy Number Variation"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": ["TSV", "TXT"]
                    }
                }
            ]
        },
        "fields": "file_id,file_name,data_type,data_format,associated_entities.entity_submitter_id,associated_entities.entity_type",
        "size": 10000
    }
    
    try:
        response = requests.post(gdc_api_url, json=query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Found {len(hits)} files")
        
        # Organize by sample and data type
        sample_data = {}
        
        for hit in hits:
            file_id = hit.get('file_id')
            data_type = hit.get('data_type')
            file_name = hit.get('file_name')
            associated_entities = hit.get('associated_entities', [])
            
            # Find the sample barcode
            sample_barcode = None
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    sample_barcode = entity_id
                    break
            
            if sample_barcode:
                if sample_barcode not in sample_data:
                    sample_data[sample_barcode] = {}
                
                if data_type == "Gene Expression Quantification":
                    sample_data[sample_barcode]['expression'] = {
                        'file_id': file_id,
                        'file_name': file_name
                    }
                elif data_type == "Copy Number Variation":
                    sample_data[sample_barcode]['cnv'] = {
                        'file_id': file_id,
                        'file_name': file_name
                    }
        
        # Find samples with both data types
        overlapping_samples = {}
        for sample, data_types in sample_data.items():
            if 'expression' in data_types and 'cnv' in data_types:
                overlapping_samples[sample] = data_types
        
        logger.info(f"Found {len(overlapping_samples)} samples with both expression and CNV data")
        
        return overlapping_samples
        
    except Exception as e:
        logger.error(f"Error querying GDC API: {e}")
        return {}

def download_file(file_id, output_dir, file_name):
    """Download a file from GDC."""
    gdc_download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
    
    try:
        response = requests.get(gdc_download_url, stream=True)
        response.raise_for_status()
        
        output_path = Path(output_dir) / file_name
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {file_name}: {e}")
        return False

def download_overlapping_data(overlapping_samples, max_samples=100):
    """Download expression and CNV data for overlapping samples."""
    logger.info(f"Downloading data for up to {max_samples} overlapping samples...")
    
    # Create output directories
    expression_dir = Path("data/raw/expression_overlapping")
    cnv_dir = Path("data/raw/cnv_overlapping")
    
    expression_dir.mkdir(parents=True, exist_ok=True)
    cnv_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    sample_list = list(overlapping_samples.items())[:max_samples]
    
    for sample_barcode, data_types in sample_list:
        logger.info(f"Processing sample {sample_barcode} ({downloaded_count + 1}/{len(sample_list)})")
        
        success = True
        
        # Download expression data
        if 'expression' in data_types:
            expr_file = data_types['expression']
            success &= download_file(
                expr_file['file_id'],
                expression_dir,
                f"{sample_barcode}_expression.tsv.gz"
            )
        
        # Download CNV data
        if 'cnv' in data_types:
            cnv_file = data_types['cnv']
            success &= download_file(
                cnv_file['file_id'],
                cnv_dir,
                f"{sample_barcode}_cnv.tsv.gz"
            )
        
        if success:
            downloaded_count += 1
        
        # Rate limiting
        time.sleep(0.1)
    
    logger.info(f"Successfully downloaded data for {downloaded_count} samples")
    return downloaded_count

def save_sample_list(overlapping_samples, output_file="overlapping_samples.json"):
    """Save the list of overlapping samples to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(overlapping_samples, f, indent=2)
    
    logger.info(f"Saved overlapping samples list to {output_file}")

def main():
    logger.info("Starting download of overlapping expression and CNV data...")
    
    # Query for overlapping samples
    overlapping_samples = query_gdc_for_overlapping_samples()
    
    if not overlapping_samples:
        logger.error("No overlapping samples found")
        return
    
    # Save the sample list
    save_sample_list(overlapping_samples)
    
    # Download the data (limit to first 100 samples for testing)
    downloaded_count = download_overlapping_data(overlapping_samples, max_samples=100)
    
    logger.info(f"Download complete! Downloaded data for {downloaded_count} samples")
    logger.info("Files saved to:")
    logger.info("  - data/raw/expression_overlapping/")
    logger.info("  - data/raw/cnv_overlapping/")

if __name__ == "__main__":
    main() 