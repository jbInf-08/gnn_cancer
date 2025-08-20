import os
import requests
import json
import time
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_patient_id(sample_id):
    """Extract patient ID from TCGA sample ID."""
    parts = sample_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return sample_id

def query_expression_samples():
    """Query for expression samples and group by patient."""
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
        
        # Group by patient ID
        patient_expression = defaultdict(list)
        
        for hit in hits:
            file_id = hit.get('file_id')
            file_name = hit.get('file_name')
            associated_entities = hit.get('associated_entities', [])
            
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    patient_id = extract_patient_id(entity_id)
                    patient_expression[patient_id].append({
                        'file_id': file_id,
                        'file_name': file_name,
                        'aliquot_id': entity_id
                    })
                    break
        
        logger.info(f"Found {len(patient_expression)} patients with expression data")
        return patient_expression
        
    except Exception as e:
        logger.error(f"Error querying expression data: {e}")
        return {}

def query_cnv_samples():
    """Query for CNV samples and group by patient."""
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
        
        # Group by patient ID
        patient_cnv = defaultdict(list)
        
        for hit in hits:
            file_id = hit.get('file_id')
            file_name = hit.get('file_name')
            associated_entities = hit.get('associated_entities', [])
            
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    patient_id = extract_patient_id(entity_id)
                    patient_cnv[patient_id].append({
                        'file_id': file_id,
                        'file_name': file_name,
                        'aliquot_id': entity_id
                    })
                    break
        
        logger.info(f"Found {len(patient_cnv)} patients with CNV data")
        return patient_cnv
        
    except Exception as e:
        logger.error(f"Error querying CNV data: {e}")
        return {}

def find_overlapping_patients(patient_expression, patient_cnv):
    """Find patients that have both expression and CNV data."""
    logger.info("Finding overlapping patients...")
    
    expression_patients = set(patient_expression.keys())
    cnv_patients = set(patient_cnv.keys())
    
    overlapping_patients = expression_patients.intersection(cnv_patients)
    
    logger.info(f"Found {len(overlapping_patients)} overlapping patients")
    
    overlapping_data = {}
    for patient_id in overlapping_patients:
        overlapping_data[patient_id] = {
            'expression': patient_expression[patient_id],
            'cnv': patient_cnv[patient_id]
        }
    
    return overlapping_data

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

def download_patient_data(overlapping_patients, max_patients=20):
    """Download expression and CNV data for overlapping patients."""
    logger.info(f"Downloading data for up to {max_patients} overlapping patients...")
    
    # Create output directories
    expression_dir = Path("data/raw/expression_patients")
    cnv_dir = Path("data/raw/cnv_patients")
    
    expression_dir.mkdir(parents=True, exist_ok=True)
    cnv_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    patient_list = list(overlapping_patients.items())[:max_patients]
    
    for patient_id, data_types in patient_list:
        logger.info(f"Processing patient {patient_id} ({downloaded_count + 1}/{len(patient_list)})")
        
        success = True
        
        # Download first expression file for this patient
        if 'expression' in data_types and data_types['expression']:
            expr_file = data_types['expression'][0]  # Take first expression file
            success &= download_file(
                expr_file['file_id'],
                expression_dir,
                f"{patient_id}_expression.tsv.gz"
            )
        
        # Download first CNV file for this patient
        if 'cnv' in data_types and data_types['cnv']:
            cnv_file = data_types['cnv'][0]  # Take first CNV file
            success &= download_file(
                cnv_file['file_id'],
                cnv_dir,
                f"{patient_id}_cnv.tsv.gz"
            )
        
        if success:
            downloaded_count += 1
        
        # Rate limiting
        time.sleep(0.2)
    
    logger.info(f"Successfully downloaded data for {downloaded_count} patients")
    return downloaded_count

def save_patient_list(overlapping_patients, output_file="overlapping_patients.json"):
    """Save the list of overlapping patients to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(overlapping_patients, f, indent=2)
    
    logger.info(f"Saved overlapping patients list to {output_file}")

def main():
    logger.info("Starting download of patient-level overlapping data...")
    
    # Query for expression samples
    patient_expression = query_expression_samples()
    
    # Query for CNV samples
    patient_cnv = query_cnv_samples()
    
    if not patient_expression or not patient_cnv:
        logger.error("Failed to query expression or CNV samples")
        return
    
    # Find overlapping patients
    overlapping_patients = find_overlapping_patients(patient_expression, patient_cnv)
    
    if not overlapping_patients:
        logger.error("No overlapping patients found")
        return
    
    # Save the patient list
    save_patient_list(overlapping_patients)
    
    # Download the data (limit to first 20 patients for testing)
    downloaded_count = download_patient_data(overlapping_patients, max_patients=20)
    
    logger.info(f"Download complete! Downloaded data for {downloaded_count} patients")
    logger.info("Files saved to:")
    logger.info("  - data/raw/expression_patients/")
    logger.info("  - data/raw/cnv_patients/")

if __name__ == "__main__":
    main() 