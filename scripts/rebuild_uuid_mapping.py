import os
import pandas as pd
import requests
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_uuids_from_data():
    """Extract UUIDs from expression and CNV data files."""
    logger.info("Extracting UUIDs from expression and CNV data...")
    
    # Get expression UUIDs
    expression_dir = "data/raw/expression"
    expression_files = list(Path(expression_dir).glob("*.tsv.gz"))
    expression_uuids = [f.stem.replace('.tsv', '') for f in expression_files]
    
    # Get CNV UUIDs
    cnv_dir = "data/raw/cnv"
    cnv_files = list(Path(cnv_dir).glob("*.tsv.gz"))
    cnv_uuids = [f.stem.replace('.tsv', '') for f in cnv_files]
    
    # Combine and get unique UUIDs
    all_uuids = list(set(expression_uuids + cnv_uuids))
    
    logger.info(f"Found {len(expression_uuids)} expression UUIDs")
    logger.info(f"Found {len(cnv_uuids)} CNV UUIDs")
    logger.info(f"Total unique UUIDs: {len(all_uuids)}")
    
    return all_uuids

def query_gdc_for_barcodes(uuids, batch_size=100):
    """Query GDC API to get TCGA barcodes for UUIDs."""
    logger.info(f"Querying GDC API for {len(uuids)} UUIDs...")
    
    # GDC API endpoint for files
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    mappings = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(uuids) + batch_size - 1)//batch_size}")
        
        # Build query for this batch
        query = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "file_id",
                    "value": batch
                }
            },
            "fields": "file_id,associated_entities.entity_submitter_id",
            "size": batch_size
        }
        
        try:
            response = requests.post(gdc_api_url, json=query)
            response.raise_for_status()
            
            data = response.json()
            
            for hit in data.get('data', {}).get('hits', []):
                file_id = hit.get('file_id')
                associated_entities = hit.get('associated_entities', [])
                
                # Look for TCGA barcode in associated entities
                for entity in associated_entities:
                    entity_id = entity.get('entity_submitter_id', '')
                    if entity_id.startswith('TCGA-'):
                        mappings.append({
                            'uuid': file_id,
                            'barcode': entity_id
                        })
                        break  # Use first TCGA barcode found
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error querying batch {i//batch_size + 1}: {e}")
            continue
    
    logger.info(f"Found {len(mappings)} UUID to barcode mappings")
    return mappings

def save_mapping(mappings, output_file="uuid_to_barcode_new.csv"):
    """Save the UUID to barcode mapping to a CSV file."""
    if not mappings:
        logger.warning("No mappings found to save")
        return
    
    df = pd.DataFrame(mappings)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(mappings)} mappings to {output_file}")
    
    # Show some sample mappings
    logger.info("Sample mappings:")
    for i, row in df.head(5).iterrows():
        logger.info(f"  {row['uuid']} -> {row['barcode']}")

def main():
    logger.info("Starting UUID to barcode mapping rebuild...")
    
    # Get UUIDs from our data
    uuids = get_uuids_from_data()
    
    if not uuids:
        logger.error("No UUIDs found in data directories")
        return
    
    # Query GDC for barcodes
    mappings = query_gdc_for_barcodes(uuids)
    
    # Save the new mapping
    save_mapping(mappings)
    
    logger.info("UUID to barcode mapping rebuild complete!")

if __name__ == "__main__":
    main() 