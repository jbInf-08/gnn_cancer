import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_gdc_data():
    """Explore available data types and formats in GDC for BRCA."""
    logger.info("Exploring GDC data for BRCA project...")
    
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    # First, let's see what data types are available
    query_data_types = {
        "filters": {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-BRCA"]
            }
        },
        "fields": "data_type,data_format,file_name",
        "size": 1000
    }
    
    try:
        response = requests.post(gdc_api_url, json=query_data_types)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Found {len(hits)} files")
        
        # Collect unique data types and formats
        data_types = set()
        data_formats = set()
        file_names = []
        
        for hit in hits:
            data_type = hit.get('data_type')
            data_format = hit.get('data_format')
            file_name = hit.get('file_name')
            
            if data_type:
                data_types.add(data_type)
            if data_format:
                data_formats.add(data_format)
            if file_name:
                file_names.append(file_name)
        
        logger.info("Available data types:")
        for dt in sorted(data_types):
            logger.info(f"  - {dt}")
        
        logger.info("Available data formats:")
        for df in sorted(data_formats):
            logger.info(f"  - {df}")
        
        logger.info("Sample file names:")
        for fn in file_names[:10]:
            logger.info(f"  - {fn}")
        
        return data_types, data_formats
        
    except Exception as e:
        logger.error(f"Error exploring GDC data: {e}")
        return set(), set()

def query_specific_data_types():
    """Query for specific data types that might have overlapping samples."""
    logger.info("Querying for specific data types...")
    
    gdc_api_url = "https://api.gdc.cancer.gov/files"
    
    # Query for RNA-Seq expression data
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
        "size": 100
    }
    
    try:
        response = requests.post(gdc_api_url, json=expression_query)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        
        logger.info(f"Found {len(hits)} expression files")
        
        # Get sample IDs from expression data
        expression_samples = set()
        for hit in hits:
            associated_entities = hit.get('associated_entities', [])
            for entity in associated_entities:
                entity_id = entity.get('entity_submitter_id', '')
                if entity_id.startswith('TCGA-'):
                    expression_samples.add(entity_id)
        
        logger.info(f"Found {len(expression_samples)} unique expression samples")
        logger.info("Sample expression samples:")
        for sample in list(expression_samples)[:5]:
            logger.info(f"  - {sample}")
        
        return expression_samples
        
    except Exception as e:
        logger.error(f"Error querying expression data: {e}")
        return set()

def main():
    logger.info("Starting GDC data exploration...")
    
    # Explore available data types
    data_types, data_formats = explore_gdc_data()
    
    # Query for specific data types
    expression_samples = query_specific_data_types()
    
    logger.info("Exploration complete!")

if __name__ == "__main__":
    main() 