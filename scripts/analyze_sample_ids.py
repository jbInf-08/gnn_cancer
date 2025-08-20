import requests
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sample_ids():
    """Analyze sample ID formats to understand the overlap issue."""
    logger.info("Analyzing sample ID formats...")
    
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    # Query for expression samples with more details
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
        "fields": "file_id,file_name,data_type,associated_entities.entity_submitter_id,associated_entities.entity_type",
        "size": 100
    }
    
    try:
        response = requests.post(gdc_api_url, json=expression_query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Analyzing {len(hits)} expression files")
        
        # Analyze entity types and IDs
        entity_types = defaultdict(set)
        sample_ids = []
        
        for hit in hits:
            associated_entities = hit.get('associated_entities', [])
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                entity_type = entity.get('entity_type', '')
                
                if entity_id:
                    entity_types[entity_type].add(entity_id)
                    if entity_id.startswith('TCGA-'):
                        sample_ids.append((entity_id, entity_type))
        
        logger.info("Entity types found in expression data:")
        for entity_type, ids in entity_types.items():
            logger.info(f"  {entity_type}: {len(ids)} unique IDs")
            logger.info(f"    Sample IDs: {list(ids)[:5]}")
        
        logger.info("Sample TCGA IDs from expression data:")
        for sample_id, entity_type in sample_ids[:10]:
            logger.info(f"  {sample_id} ({entity_type})")
        
        return entity_types, sample_ids
        
    except Exception as e:
        logger.error(f"Error analyzing expression data: {e}")
        return {}, []

def analyze_cnv_sample_ids():
    """Analyze CNV sample ID formats."""
    logger.info("Analyzing CNV sample ID formats...")
    
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
        "fields": "file_id,file_name,data_type,associated_entities.entity_submitter_id,associated_entities.entity_type",
        "size": 100
    }
    
    try:
        response = requests.post(gdc_api_url, json=cnv_query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Analyzing {len(hits)} CNV files")
        
        # Analyze entity types and IDs
        entity_types = defaultdict(set)
        sample_ids = []
        
        for hit in hits:
            associated_entities = hit.get('associated_entities', [])
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                entity_type = entity.get('entity_type', '')
                
                if entity_id:
                    entity_types[entity_type].add(entity_id)
                    if entity_id.startswith('TCGA-'):
                        sample_ids.append((entity_id, entity_type))
        
        logger.info("Entity types found in CNV data:")
        for entity_type, ids in entity_types.items():
            logger.info(f"  {entity_type}: {len(ids)} unique IDs")
            logger.info(f"    Sample IDs: {list(ids)[:5]}")
        
        logger.info("Sample TCGA IDs from CNV data:")
        for sample_id, entity_type in sample_ids[:10]:
            logger.info(f"  {sample_id} ({entity_type})")
        
        return entity_types, sample_ids
        
    except Exception as e:
        logger.error(f"Error analyzing CNV data: {e}")
        return {}, []

def extract_patient_ids(sample_ids):
    """Extract patient IDs from sample IDs."""
    patient_ids = set()
    
    for sample_id, entity_type in sample_ids:
        # TCGA format: TCGA-XX-XXXX-XX-XX-XXXX-XX
        # Patient ID is: TCGA-XX-XXXX
        parts = sample_id.split('-')
        if len(parts) >= 3:
            patient_id = '-'.join(parts[:3])
            patient_ids.add(patient_id)
    
    return patient_ids

def main():
    logger.info("Starting sample ID analysis...")
    
    # Analyze expression sample IDs
    expr_entity_types, expr_sample_ids = analyze_sample_ids()
    
    # Analyze CNV sample IDs
    cnv_entity_types, cnv_sample_ids = analyze_cnv_sample_ids()
    
    # Extract patient IDs
    expr_patient_ids = extract_patient_ids(expr_sample_ids)
    cnv_patient_ids = extract_patient_ids(cnv_sample_ids)
    
    logger.info(f"Expression patient IDs: {len(expr_patient_ids)}")
    logger.info(f"CNV patient IDs: {len(cnv_patient_ids)}")
    
    # Check overlap at patient level
    patient_overlap = expr_patient_ids.intersection(cnv_patient_ids)
    logger.info(f"Patient-level overlap: {len(patient_overlap)}")
    
    if patient_overlap:
        logger.info("Sample overlapping patient IDs:")
        for patient_id in list(patient_overlap)[:5]:
            logger.info(f"  {patient_id}")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 