import os
import json
import requests
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

def setup_directories():
    """Create necessary directories if they don't exist."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path('data/raw/expression').mkdir(parents=True, exist_ok=True)
    Path('data/raw/cnv').mkdir(parents=True, exist_ok=True)
    Path('data/raw/clinical').mkdir(parents=True, exist_ok=True)

def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_file_ids(session, project_id, data_category, data_type, max_files=1000):
    """Get file IDs from GDC API with configurable max files."""
    files_endpoint = "https://api.gdc.cancer.gov/files"
    
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_id
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_category",
                    "value": data_category
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_type",
                    "value": data_type
                }
            }
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "format": "JSON",
        "size": str(max_files)
    }
    
    response = session.get(files_endpoint, params=params)
    response.raise_for_status()
    
    data = response.json()
    return [hit["file_id"] for hit in data["data"]["hits"]]

def download_file(session, file_id, output_path):
    """Download a file from GDC API."""
    print(f"Downloading file {file_id}...")
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = session.get(
                f"https://api.gdc.cancer.gov/data/{file_id}",
                stream=True
            )
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded to {output_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to download after {max_retries} attempts")
                return False

def download_clinical_data(session, project_id):
    """Download comprehensive clinical data from GDC."""
    print("\nDownloading clinical data...")
    
    try:
        # GDC clinical data endpoint
        url = "https://api.gdc.cancer.gov/cases"
        params = {
            "filters": json.dumps({
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_id
                }
            }),
            "fields": "cases.case_id,cases.demographic.gender,cases.demographic.race,cases.demographic.ethnicity,cases.diagnoses.age_at_diagnosis,cases.diagnoses.vital_status,cases.diagnoses.days_to_death,cases.diagnoses.days_to_last_follow_up,cases.diagnoses.tumor_stage,cases.diagnoses.diagnosis,cases.samples.sample_type",
            "format": "TSV",
            "size": "1000"
        }
        
        response = session.get(url, params=params)
        response.raise_for_status()
        
        clinical_path = Path("data/raw/clinical/clinical_data.tsv")
        with open(clinical_path, 'w') as f:
            f.write(response.text)
        
        print(f"Downloaded clinical data to {clinical_path}")
        return True
        
    except Exception as e:
        print(f"Failed to download clinical data: {e}")
        return False

def get_available_patients():
    """Get the number of available patients from mapping file."""
    mapping_file = Path("uuid_to_barcode.csv")
    if mapping_file.exists():
        mapping_df = pd.read_csv(mapping_file)
        return len(mapping_df)
    return 0

def main():
    # Setup directories
    setup_directories()

    # Create session with retry logic
    session = create_session()

    # Project and data type specifications
    project_id = "TCGA-BRCA"
    
    # Get available patient count
    available_patients = get_available_patients()
    print(f"Available patients in mapping: {available_patients}")
    
    # Target: Download data for up to 358 patients (or all available)
    target_patients = min(358, available_patients)
    print(f"Target: Download data for {target_patients} patients")

    # Download clinical data first
    download_clinical_data(session, project_id)

    # Download gene expression data (all files)
    print("\nGetting gene expression file IDs...")
    expr_file_ids = get_file_ids(
        session,
        project_id,
        "Transcriptome Profiling",
        "Gene Expression Quantification",
        max_files=target_patients * 2  # Allow for multiple files per patient
    )
    if expr_file_ids:
        print(f"Found {len(expr_file_ids)} expression files")
        for file_id in expr_file_ids:
            expr_path = Path(f"data/raw/expression/{file_id}.tsv.gz")
            if not expr_path.exists():
                download_file(session, file_id, expr_path)
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"File {expr_path} already exists, skipping...")
    else:
        print("No expression files found")

    # Download CNV data (all files)
    print("\nGetting CNV file IDs...")
    cnv_file_ids = get_file_ids(
        session,
        project_id,
        "Copy Number Variation",
        "Copy Number Segment",
        max_files=target_patients * 2
    )
    if cnv_file_ids:
        print(f"Found {len(cnv_file_ids)} CNV files")
        for file_id in cnv_file_ids:
            cnv_path = Path(f"data/raw/cnv/{file_id}.tsv.gz")
            if not cnv_path.exists():
                download_file(session, file_id, cnv_path)
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"File {cnv_path} already exists, skipping...")
    else:
        print("No CNV files found")

    # Download mutation data (multiple files for better coverage)
    print("\nGetting mutation file IDs...")
    mut_file_ids = get_file_ids(
        session,
        project_id,
        "Simple Nucleotide Variation",
        "Masked Somatic Mutation",
        max_files=target_patients
    )
    if mut_file_ids:
        print(f"Found {len(mut_file_ids)} mutation files")
        # Download first few mutation files for better coverage
        for i, file_id in enumerate(mut_file_ids[:5]):  # Download up to 5 mutation files
            mut_path = Path(f"data/raw/BRCA_mutation_{i+1}.maf.gz")
            if not mut_path.exists():
                download_file(session, file_id, mut_path)
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"File {mut_path} already exists, skipping...")
    else:
        print("No mutation files found")

    # Download gene-level CNV data (all files)
    print("\nGetting gene-level CNV file IDs...")
    # Try different data type combinations for gene-level CNV
    gene_cnv_file_ids = get_file_ids(
        session,
        project_id,
        "Copy Number Variation",
        "Copy Number Segment",
        max_files=target_patients * 2
    )
    if not gene_cnv_file_ids:
        # Try alternative data type
        gene_cnv_file_ids = get_file_ids(
            session,
            project_id,
            "Copy Number Variation",
            "Gene Level Copy Number",
            max_files=target_patients * 2
        )
    if gene_cnv_file_ids:
        print(f"Found {len(gene_cnv_file_ids)} gene-level CNV files")
        Path('data/raw/cnv_gene').mkdir(parents=True, exist_ok=True)
        for file_id in gene_cnv_file_ids:
            out_path = f"data/raw/cnv_gene/{file_id}.tsv.gz"
            if not os.path.exists(out_path):
                download_file(session, file_id, out_path)
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"File {out_path} already exists, skipping...")
    else:
        print("No gene-level CNV files found with any data type combination.")

    print(f"\nDownload complete. Target: {target_patients} patients")

if __name__ == "__main__":
    main() 