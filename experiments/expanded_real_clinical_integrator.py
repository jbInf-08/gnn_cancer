import os
import requests
import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gzip
import tarfile
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExpandedRealClinicalIntegrator:
    """
    Expanded real clinical data integrator that fetches large-scale datasets
    to far exceed paper results
    """
    
    def __init__(self, output_dir: str = "data/expanded_real_clinical"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "mutations").mkdir(exist_ok=True)
        (self.output_dir / "expression").mkdir(exist_ok=True)
        (self.output_dir / "clinical").mkdir(exist_ok=True)
        (self.output_dir / "protein").mkdir(exist_ok=True)
        (self.output_dir / "cnv").mkdir(exist_ok=True)
        (self.output_dir / "methylation").mkdir(exist_ok=True)
        (self.output_dir / "mirna").mkdir(exist_ok=True)
        
        # API endpoints
        self.gdc_api_base = "https://api.gdc.cancer.gov"
        self.tcga_data_portal = "https://portal.gdc.cancer.gov"
        self.cptac_data_portal = "https://cptac-data-portal.georgetown.edu"
        
        # Session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Expanded cancer types to fetch
        self.cancer_types = [
            'BRCA', 'LUAD', 'LUSC', 'COAD', 'READ', 'STAD', 'LIHC', 'KIRC',
            'KIRP', 'THCA', 'PRAD', 'BLCA', 'HNSC', 'CESC', 'UCEC', 'OV',
            'SKCM', 'DLBC', 'LAML', 'ACC', 'CHOL', 'ESCA', 'GBM', 'LGG',
            'MESO', 'PAAD', 'PCPG', 'SARC', 'TGCT', 'THYM', 'UCS', 'UVM'
        ]
        
        # Data types to fetch
        self.data_types = {
            'mutations': ['Simple Nucleotide Variation'],
            'expression': ['Gene Expression Quantification'],
            'clinical': ['Clinical Supplement'],
            'protein': ['Protein Expression Quantification'],
            'cnv': ['Copy Number Variation'],
            'methylation': ['Methylation Beta Value'],
            'mirna': ['Isoform Expression Quantification']
        }
    
    def fetch_tcga_file_manifest(self, cancer_type: str, data_type: str, max_files: int = 500) -> List[Dict]:
        """
        Fetch file manifest from TCGA GDC API with expanded limits
        """
        logger.info(f"Fetching {data_type} manifest for {cancer_type}")
        
        # GDC API query
        query = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "=",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": f"TCGA-{cancer_type}"
                        }
                    },
                    {
                        "op": "=",
                        "content": {
                            "field": "files.data_type",
                            "value": data_type
                        }
                    },
                    {
                        "op": "=",
                        "content": {
                            "field": "files.experimental_strategy",
                            "value": "WXS" if data_type == "Simple Nucleotide Variation" else "RNA-Seq"
                        }
                    }
                ]
            },
            "format": "JSON",
            "size": max_files
        }
        
        try:
            response = self.session.post(
                f"{self.gdc_api_base}/files",
                json=query,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('hits', [])
            
            logger.info(f"Found {len(files)} {data_type} files for {cancer_type}")
            return files
            
        except Exception as e:
            logger.error(f"Error fetching manifest for {cancer_type} {data_type}: {e}")
            return []
    
    def download_tcga_file(self, file_id: str, file_name: str, data_type: str) -> Optional[str]:
        """
        Download a single file from TCGA GDC
        """
        try:
            # Get file download URL
            response = self.session.get(f"{self.gdc_api_base}/data/{file_id}", timeout=60)
            response.raise_for_status()
            
            # Save file
            output_path = self.output_dir / data_type / f"{file_name}.tsv"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded {file_name} to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading {file_id}: {e}")
            return None
    
    def fetch_expanded_mutation_data(self, cancer_type: str, max_files: int = 100) -> List[str]:
        """
        Fetch expanded mutation data from TCGA
        """
        logger.info(f"Fetching expanded mutation data for {cancer_type}")
        
        # Get file manifest
        files = self.fetch_tcga_file_manifest(cancer_type, "Simple Nucleotide Variation", max_files)
        
        if not files:
            logger.warning(f"No mutation files found for {cancer_type}")
            return []
        
        # Download files
        downloaded_files = []
        for i, file_info in enumerate(files[:max_files]):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            
            logger.info(f"Downloading mutation file {i+1}/{min(len(files), max_files)}: {file_name}")
            
            file_path = self.download_tcga_file(file_id, f"{cancer_type}_mutation_{i}", "mutations")
            if file_path:
                downloaded_files.append(file_path)
            
            time.sleep(0.5)  # Rate limiting
        
        return downloaded_files
    
    def fetch_expanded_expression_data(self, cancer_type: str, max_files: int = 100) -> List[str]:
        """
        Fetch expanded expression data from TCGA
        """
        logger.info(f"Fetching expanded expression data for {cancer_type}")
        
        # Get file manifest
        files = self.fetch_tcga_file_manifest(cancer_type, "Gene Expression Quantification", max_files)
        
        if not files:
            logger.warning(f"No expression files found for {cancer_type}")
            return []
        
        # Download files
        downloaded_files = []
        for i, file_info in enumerate(files[:max_files]):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            
            logger.info(f"Downloading expression file {i+1}/{min(len(files), max_files)}: {file_name}")
            
            file_path = self.download_tcga_file(file_id, f"{cancer_type}_expression_{i}", "expression")
            if file_path:
                downloaded_files.append(file_path)
            
            time.sleep(0.5)  # Rate limiting
        
        return downloaded_files
    
    def fetch_expanded_clinical_data(self, cancer_type: str, max_files: int = 50) -> List[str]:
        """
        Fetch expanded clinical data from TCGA
        """
        logger.info(f"Fetching expanded clinical data for {cancer_type}")
        
        # Get file manifest
        files = self.fetch_tcga_file_manifest(cancer_type, "Clinical Supplement", max_files)
        
        if not files:
            logger.warning(f"No clinical files found for {cancer_type}")
            return []
        
        # Download files
        downloaded_files = []
        for i, file_info in enumerate(files[:max_files]):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            
            logger.info(f"Downloading clinical file {i+1}/{min(len(files), max_files)}: {file_name}")
            
            file_path = self.download_tcga_file(file_id, f"{cancer_type}_clinical_{i}", "clinical")
            if file_path:
                downloaded_files.append(file_path)
            
            time.sleep(0.5)  # Rate limiting
        
        return downloaded_files
    
    def fetch_expanded_protein_data(self, cancer_type: str) -> List[str]:
        """
        Fetch expanded protein data from CPTAC
        """
        logger.info(f"Fetching expanded protein data for {cancer_type}")
        
        # CPTAC data URLs for expanded cancer types
        cptac_urls = {
            'BRCA': {
                'protein': 'https://cptac-data-portal.georgetown.edu/cptac/data/BRCA/Proteome',
                'clinical': 'https://cptac-data-portal.georgetown.edu/cptac/data/BRCA/Clinical'
            },
            'COAD': {
                'protein': 'https://cptac-data-portal.georgetown.edu/cptac/data/COAD/Proteome',
                'clinical': 'https://cptac-data-portal.georgetown.edu/cptac/data/COAD/Clinical'
            },
            'LUAD': {
                'protein': 'https://cptac-data-portal.georgetown.edu/cptac/data/LUAD/Proteome',
                'clinical': 'https://cptac-data-portal.georgetown.edu/cptac/data/LUAD/Clinical'
            },
            'OV': {
                'protein': 'https://cptac-data-portal.georgetown.edu/cptac/data/OV/Proteome',
                'clinical': 'https://cptac-data-portal.georgetown.edu/cptac/data/OV/Clinical'
            }
        }
        
        if cancer_type not in cptac_urls:
            logger.warning(f"No CPTAC URLs configured for {cancer_type}")
            return []
        
        downloaded_files = []
        
        try:
            # Download protein data
            protein_url = cptac_urls[cancer_type]['protein']
            response = self.session.get(protein_url, timeout=60)
            
            if response.status_code == 200:
                output_path = self.output_dir / "protein" / f"{cancer_type}_cptac_protein.tsv"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(str(output_path))
                logger.info(f"Downloaded CPTAC protein data for {cancer_type}")
            else:
                logger.warning(f"Could not download CPTAC protein data for {cancer_type}")
        
        except Exception as e:
            logger.error(f"Error downloading CPTAC protein data for {cancer_type}: {e}")
        
        return downloaded_files
    
    def fetch_expanded_string_ppi_network(self, genes: List[str] = None) -> str:
        """
        Fetch expanded PPI network from STRING database
        """
        logger.info("Fetching expanded PPI network from STRING")
        
        if genes is None:
            # Expanded cancer genes list
            genes = [
                'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 'KRAS', 'NRAS', 'BRAF',
                'EGFR', 'ERBB2', 'MYC', 'CDKN2A', 'RB1', 'APC', 'VHL', 'NF1',
                'ATM', 'CHEK2', 'PALB2', 'BARD1', 'RAD51C', 'RAD51D', 'BRIP1',
                'CDH1', 'STK11', 'SMAD4', 'TGFBR2', 'MSH2', 'MLH1', 'MSH6',
                'PMS2', 'EPCAM', 'MUTYH', 'NTHL1', 'POLE', 'POLD1', 'ARID1A',
                'CTNNB1', 'FBXW7', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
                'KMT2D', 'KMT2C', 'CREBBP', 'EP300', 'ARID2', 'SMARCA4',
                'SMARCB1', 'SMARCD1', 'SMARCE1', 'SMARCC1', 'SMARCC2',
                'ARID1B', 'SMARCA2', 'SMARCA1', 'SMARCB1', 'SMARCD2',
                'EP400', 'BRD7', 'BRD9', 'BRD4', 'BRD2', 'BRD3',
                'CDK4', 'CDK6', 'CCND1', 'CCND2', 'CCND3', 'CCNE1',
                'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'E2F6',
                'MDM2', 'MDM4', 'BAX', 'BAK1', 'BCL2', 'BCL2L1',
                'CASP3', 'CASP8', 'CASP9', 'FAS', 'FASLG', 'TNF',
                'TNFRSF10A', 'TNFRSF10B', 'TNFRSF10C', 'TNFRSF10D'
            ]
        
        try:
            # STRING API call with expanded parameters
            string_url = "https://string-db.org/api/tsv/network"
            
            params = {
                'identifiers': '%0d'.join(genes),
                'species': 9606,  # Human
                'required_score': 600,  # Lower threshold for more interactions
                'network_type': 'physical'
            }
            
            response = self.session.get(string_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Save PPI data
            ppi_file = self.output_dir / "string_ppi_network.tsv"
            with open(ppi_file, 'w') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded expanded STRING PPI network with {len(response.text.splitlines())} interactions")
            return str(ppi_file)
            
        except Exception as e:
            logger.error(f"Error fetching STRING PPI data: {e}")
            return None
    
    def fetch_expanded_kegg_pathways(self) -> str:
        """
        Fetch expanded pathway data from KEGG
        """
        logger.info("Fetching expanded pathway data from KEGG")
        
        try:
            # KEGG pathway data for cancer and related pathways
            kegg_url = "https://rest.kegg.org/list/pathway/hsa"
            response = self.session.get(kegg_url, timeout=60)
            response.raise_for_status()
            
            # Parse KEGG pathways - expanded to include more pathways
            pathways = []
            for line in response.text.strip().split('\n'):
                if any(keyword in line.lower() for keyword in ['cancer', 'tumor', 'apoptosis', 'cell cycle', 'dna repair', 'metabolism']):
                    pathway_id = line.split('\t')[0]
                    pathway_name = line.split('\t')[1]
                    pathways.append((pathway_id, pathway_name))
            
            # Save pathway data
            pathway_file = self.output_dir / "kegg_expanded_pathways.tsv"
            with open(pathway_file, 'w') as f:
                f.write("pathway_id\tpathway_name\n")
                for pathway_id, pathway_name in pathways:
                    f.write(f"{pathway_id}\t{pathway_name}\n")
            
            logger.info(f"Downloaded expanded KEGG pathway data with {len(pathways)} pathways")
            return str(pathway_file)
            
        except Exception as e:
            logger.error(f"Error fetching KEGG pathway data: {e}")
            return None
    
    def fetch_expanded_cnv_data(self, cancer_type: str, max_files: int = 50) -> List[str]:
        """
        Fetch expanded CNV data from TCGA
        """
        logger.info(f"Fetching expanded CNV data for {cancer_type}")
        
        # Get file manifest
        files = self.fetch_tcga_file_manifest(cancer_type, "Copy Number Variation", max_files)
        
        if not files:
            logger.warning(f"No CNV files found for {cancer_type}")
            return []
        
        # Download files
        downloaded_files = []
        for i, file_info in enumerate(files[:max_files]):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            
            logger.info(f"Downloading CNV file {i+1}/{min(len(files), max_files)}: {file_name}")
            
            file_path = self.download_tcga_file(file_id, f"{cancer_type}_cnv_{i}", "cnv")
            if file_path:
                downloaded_files.append(file_path)
            
            time.sleep(0.5)  # Rate limiting
        
        return downloaded_files
    
    def fetch_expanded_methylation_data(self, cancer_type: str, max_files: int = 50) -> List[str]:
        """
        Fetch expanded methylation data from TCGA
        """
        logger.info(f"Fetching expanded methylation data for {cancer_type}")
        
        # Get file manifest
        files = self.fetch_tcga_file_manifest(cancer_type, "Methylation Beta Value", max_files)
        
        if not files:
            logger.warning(f"No methylation files found for {cancer_type}")
            return []
        
        # Download files
        downloaded_files = []
        for i, file_info in enumerate(files[:max_files]):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            
            logger.info(f"Downloading methylation file {i+1}/{min(len(files), max_files)}: {file_name}")
            
            file_path = self.download_tcga_file(file_id, f"{cancer_type}_methylation_{i}", "methylation")
            if file_path:
                downloaded_files.append(file_path)
            
            time.sleep(0.5)  # Rate limiting
        
        return downloaded_files
    
    def create_expanded_data_summary(self) -> Dict:
        """
        Create a summary of all downloaded expanded real data
        """
        summary = {
            'total_files': 0,
            'data_types': {},
            'cancer_types': set(),
            'patients': set(),
            'mutations': 0,
            'expressions': 0,
            'clinical_records': 0,
            'protein_files': 0,
            'cnv_files': 0,
            'methylation_files': 0
        }
        
        # Count files by type
        for data_type in ['mutations', 'expression', 'clinical', 'protein', 'cnv', 'methylation']:
            dir_path = self.output_dir / data_type
            if dir_path.exists():
                files = list(dir_path.glob('*.tsv'))
                summary['data_types'][data_type] = len(files)
                summary['total_files'] += len(files)
                
                # Extract cancer types and patients from filenames
                for file in files:
                    filename = file.stem
                    if '_' in filename:
                        cancer_type = filename.split('_')[0]
                        summary['cancer_types'].add(cancer_type)
                        
                        # Extract patient ID if available
                        if 'mutation' in filename or 'expression' in filename:
                            parts = filename.split('_')
                            if len(parts) >= 3:
                                patient_id = parts[-1]
                                summary['patients'].add(patient_id)
        
        summary['cancer_types'] = list(summary['cancer_types'])
        summary['patients'] = list(summary['patients'])
        
        logger.info(f"Expanded real data summary: {summary}")
        return summary
    
    def integrate_all_expanded_real_data(self, max_files_per_type: int = 50) -> Dict:
        """
        Integrate all expanded real clinical data from multiple sources
        """
        logger.info("Starting comprehensive expanded real clinical data integration")
        
        results = {
            'mutations': [],
            'expression': [],
            'clinical': [],
            'protein': [],
            'cnv': [],
            'methylation': [],
            'ppi_network': None,
            'pathways': None
        }
        
        # Fetch data for each cancer type
        for cancer_type in self.cancer_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {cancer_type}")
            logger.info(f"{'='*60}")
            
            # Fetch mutation data
            mutation_files = self.fetch_expanded_mutation_data(cancer_type, max_files_per_type)
            results['mutations'].extend(mutation_files)
            
            # Fetch expression data
            expression_files = self.fetch_expanded_expression_data(cancer_type, max_files_per_type)
            results['expression'].extend(expression_files)
            
            # Fetch clinical data
            clinical_files = self.fetch_expanded_clinical_data(cancer_type, max_files_per_type)
            results['clinical'].extend(clinical_files)
            
            # Fetch protein data (CPTAC)
            protein_files = self.fetch_expanded_protein_data(cancer_type)
            results['protein'].extend(protein_files)
            
            # Fetch CNV data
            cnv_files = self.fetch_expanded_cnv_data(cancer_type, max_files_per_type)
            results['cnv'].extend(cnv_files)
            
            # Fetch methylation data
            methylation_files = self.fetch_expanded_methylation_data(cancer_type, max_files_per_type)
            results['methylation'].extend(methylation_files)
            
            time.sleep(2)  # Rate limiting between cancer types
        
        # Fetch network data
        logger.info("\nFetching expanded network data...")
        results['ppi_network'] = self.fetch_expanded_string_ppi_network()
        results['pathways'] = self.fetch_expanded_kegg_pathways()
        
        # Create summary
        summary = self.create_expanded_data_summary()
        
        # Save summary
        summary_file = self.output_dir / "expanded_real_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\nExpanded real data integration completed!")
        logger.info(f"Total files downloaded: {summary['total_files']}")
        logger.info(f"Cancer types: {summary['cancer_types']}")
        logger.info(f"Estimated patients: {len(summary['patients'])}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return results

def main():
    """Main function to integrate expanded real clinical data"""
    logger.info("Starting expanded real clinical data integration")
    
    integrator = ExpandedRealClinicalIntegrator()
    
    # Integrate all expanded real data
    results = integrator.integrate_all_expanded_real_data()
    
    logger.info("Expanded real clinical data integration completed successfully!")
    logger.info(f"Results saved to: {integrator.output_dir}")

if __name__ == "__main__":
    main() 