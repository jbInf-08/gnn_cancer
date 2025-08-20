import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gzip
import json
from tqdm import tqdm
import subprocess
import shutil
from datetime import datetime
import tarfile
import zipfile
import ftplib

class CancerDataDownloader:
    def __init__(self, data_dir='data/raw', tcga_file_ids=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # TCGA file UUIDs (user should provide real ones)
        # Example: {'mutation': ['uuid1', 'uuid2'], ...}
        self.tcga_file_ids = tcga_file_ids or {
            'mutation': [],
            'expression': [],
            'cnv': []
        }

    def download_tcga_data(self, cancer_type='BRCA', data_type='mutation'):
        """Download data from TCGA using the GDC Data Portal API.
        Args:
            cancer_type (str): e.g. 'BRCA'
            data_type (str): 'mutation', 'expression', or 'cnv'
        Note:
            - You must provide real file UUIDs in self.tcga_file_ids[data_type].
            - Get UUIDs from https://portal.gdc.cancer.gov/
        """
        try:
            self.logger.info(f"Downloading {data_type} data for {cancer_type} from TCGA")
            cancer_dir = self.data_dir / 'tcga' / cancer_type / data_type
            cancer_dir.mkdir(parents=True, exist_ok=True)
            base_url = "https://api.gdc.cancer.gov/data"
            file_ids = self.tcga_file_ids.get(data_type, [])
            if not file_ids:
                self.logger.warning(f"No file UUIDs provided for {data_type}. Please update tcga_file_ids.")
            for file_id in tqdm(file_ids, desc=f"Downloading {data_type} files"):
                try:
                    response = requests.get(f"{base_url}/{file_id}")
                    if response.status_code == 200:
                        output_path = cancer_dir / f"{file_id}.gz"
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        self.logger.info(f"Successfully downloaded file {file_id}")
                    else:
                        self.logger.error(f"Failed to download file {file_id}: {response.status_code}")
                except Exception as e:
                    self.logger.error(f"Error downloading file {file_id}: {str(e)}")
            self.logger.info(f"Completed downloading TCGA {cancer_type} {data_type} data")
        except Exception as e:
            self.logger.error(f"Error downloading TCGA data: {str(e)}")
            raise

    def download_clinvar_data(self):
        """Download data from ClinVar using FTP."""
        try:
            self.logger.info("Downloading data from ClinVar (FTP)")
            clinvar_dir = self.data_dir / 'clinvar'
            clinvar_dir.mkdir(parents=True, exist_ok=True)
            ftp_host = "ftp.ncbi.nlm.nih.gov"
            ftp_dir = "/pub/clinvar/"
            files = [
                'variant_summary.txt.gz',
                'gene_condition_source_id.txt.gz'
            ]
            with ftplib.FTP(ftp_host) as ftp:
                ftp.login()
                ftp.cwd(ftp_dir)
                for file in tqdm(files, desc="Downloading ClinVar files"):
                    try:
                        output_path = clinvar_dir / file
                        with open(output_path, 'wb') as f:
                            ftp.retrbinary(f"RETR {file}", f.write)
                        self.logger.info(f"Successfully downloaded {file}")
                    except Exception as e:
                        self.logger.error(f"Error downloading {file}: {str(e)}")
            self.logger.info("Completed downloading ClinVar data")
        except Exception as e:
            self.logger.error(f"Error downloading ClinVar data: {str(e)}")
            raise

    def download_cosmic_data(self):
        """Download data from COSMIC using their API."""
        try:
            self.logger.info("Downloading data from COSMIC")
            cosmic_dir = self.data_dir / 'cosmic'
            cosmic_dir.mkdir(parents=True, exist_ok=True)
            base_url = "https://cancer.sanger.ac.uk/cosmic/api/v1"
            endpoints = {
                'mutations': '/mutations',
                'genes': '/genes',
                'census': '/census'
            }
            for data_type, endpoint in endpoints.items():
                try:
                    response = requests.get(f"{base_url}{endpoint}")
                    if response.status_code == 200:
                        output_path = cosmic_dir / f"{data_type}.json"
                        with open(output_path, 'w') as f:
                            json.dump(response.json(), f)
                        self.logger.info(f"Successfully downloaded {data_type} data")
                    else:
                        self.logger.error(f"Failed to download {data_type} data: {response.status_code}")
                except Exception as e:
                    self.logger.error(f"Error downloading {data_type} data: {str(e)}")
            self.logger.info("Completed downloading COSMIC data")
        except Exception as e:
            self.logger.error(f"Error downloading COSMIC data: {str(e)}")
            raise

    def download_all_data(self, cancer_type='BRCA'):
        """Download data from all available sources."""
        try:
            self.logger.info("Starting comprehensive data download")
            self.download_tcga_data(cancer_type, 'mutation')
            self.download_tcga_data(cancer_type, 'expression')
            self.download_tcga_data(cancer_type, 'cnv')
            self.download_cosmic_data()
            self.download_clinvar_data()
            self.logger.info("Successfully completed comprehensive data download")
        except Exception as e:
            self.logger.error(f"Error in comprehensive data download: {str(e)}")
            raise

def main():
    # Example: update these with real UUIDs from GDC Data Portal
    tcga_file_ids = {
        'mutation': ['REAL-UUID-1', 'REAL-UUID-2'],
        'expression': ['REAL-UUID-3'],
        'cnv': ['REAL-UUID-4']
    }
    downloader = CancerDataDownloader(tcga_file_ids=tcga_file_ids)
    downloader.download_all_data()

if __name__ == "__main__":
    main() 