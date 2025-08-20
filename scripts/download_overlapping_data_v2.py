import os
import requests
import json
import time
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_expression_samples():
    """Query for expression samples."""
    logger.info("Querying for expression samples...")
    
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    expression_query = {
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
                        "value": ["Gene Expression Quantification"]
                    }
                }
            ]
        },
        "fields": "file_id,file_name,data_type,associated_entities.entity_submitter_id",
        "size": 1000
    }
    
    try:
        response = requests.post(gdc_api_url, json=expression_query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Found {len(hits)} expression files")
        
        expression_samples = {}
        for hit in hits:
            file_id = hit.get('file_id')
            file_name = hit.get('file_name')
            associated_entities = hit.get('associated_entities', [])
            
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    expression_samples[entity_id] = {
                        'file_id': file_id,
                        'file_name': file_name
                    }
                    break
        
        logger.info(f"Found {len(expression_samples)} unique expression samples")
        return expression_samples
        
    except Exception as e:
        logger.error(f"Error querying expression data: {e}")
        return {}

def query_cnv_samples():
    """Query for CNV samples."""
    logger.info("Querying for CNV samples...")
    
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    cnv_query = {
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
                        "value": ["Copy Number Segment"]
                    }
                }
            ]
        },
        "fields": "file_id,file_name,data_type,associated_entities.entity_submitter_id",
        "size": 1000
    }
    
    try:
        response = requests.post(gdc_api_url, json=cnv_query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Found {len(hits)} CNV files")
        
        cnv_samples = {}
        for hit in hits:
            file_id = hit.get('file_id')
            file_name = hit.get('file_name')
            associated_entities = hit.get('associated_entities', [])
            
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    cnv_samples[entity_id] = {
                        'file_id': file_id,
                        'file_name': file_name
                    }
                    break
        
        logger.info(f"Found {len(cnv_samples)} unique CNV samples")
        return cnv_samples
        
    except Exception as e:
        logger.error(f"Error querying CNV data: {e}")
        return {}

def find_overlapping_samples(expression_samples, cnv_samples):
    """Find samples that have both expression and CNV data."""
    logger.info("Finding overlapping samples...")
    
    expression_sample_ids = set(expression_samples.keys())
    cnv_sample_ids = set(cnv_samples.keys())
    
    overlapping_ids = expression_sample_ids.intersection(cnv_sample_ids)
    
    logger.info(f"Found {len(overlapping_ids)} overlapping samples")
    
    overlapping_samples = {}
    for sample_id in overlapping_ids:
        overlapping_samples[sample_id] = {
            'expression': expression_samples[sample_id],
            'cnv': cnv_samples[sample_id]
        }
    
    return overlapping_samples

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

def download_overlapping_data(overlapping_samples, max_samples=50):
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
        time.sleep(0.2)
    
    logger.info(f"Successfully downloaded data for {downloaded_count} samples")
    return downloaded_count

def save_sample_list(overlapping_samples, output_file="overlapping_samples_v2.json"):
    """Save the list of overlapping samples to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(overlapping_samples, f, indent=2)
    
    logger.info(f"Saved overlapping samples list to {output_file}")

def main():
    logger.info("Starting download of overlapping expression and CNV data (v2)...")
    
    # Query for expression samples
    expression_samples = query_expression_samples()
    
    # Query for CNV samples
    cnv_samples = query_cnv_samples()
    
    if not expression_samples or not cnv_samples:
        logger.error("Failed to query expression or CNV samples")
        return
    
    # Find overlapping samples
    overlapping_samples = find_overlapping_samples(expression_samples, cnv_samples)
    
    if not overlapping_samples:
        logger.error("No overlapping samples found")
        return
    
    # Save the sample list
    save_sample_list(overlapping_samples)
    
    # Download the data (limit to first 50 samples for testing)
    downloaded_count = download_overlapping_data(overlapping_samples, max_samples=50)
    
    logger.info(f"Download complete! Downloaded data for {downloaded_count} samples")
    logger.info("Files saved to:")
    logger.info("  - data/raw/expression_overlapping/")
    logger.info("  - data/raw/cnv_overlapping/")

if __name__ == "__main__":
    main() 