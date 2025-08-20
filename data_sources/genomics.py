import requests
import ftplib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import gzip
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo

class TCGADownloader(BaseDataSource):
    """Downloader for The Cancer Genome Atlas (TCGA) data."""
    
    @property
    def source_name(self) -> str:
        return "tcga"
    
    def authenticate(self) -> bool:
        """TCGA requires authentication for controlled data."""
        token = self._get_credential("GDC_TOKEN")
        if not token:
            self.logger.warning("No GDC token found. Some datasets may be inaccessible.")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available TCGA datasets."""
        try:
            response = requests.get("https://api.gdc.cancer.gov/projects")
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch TCGA projects: {response.status_code}")
                return []
            
            projects = response.json()["data"]["hits"]
            datasets = []
            
            for project in projects:
                datasets.append(DatasetInfo(
                    name=project["project_id"],
                    description=project.get("name", ""),
                    data_type=DataType.GENOMICS,
                    requires_auth=project.get("disease_type") in ["Controlled", "Protected"],
                    url=f"https://portal.gdc.cancer.gov/projects/{project['project_id']}"
                ))
            
            return datasets
        except Exception as e:
            self.logger.error(f"Error listing TCGA datasets: {str(e)}")
            return []
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a TCGA dataset."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        # Get file manifest
        manifest_url = f"https://api.gdc.cancer.gov/files?filters={{\"op\":\"and\",\"content\":[{{\"op\":\"=\",\"content\":{{\"field\":\"cases.project.project_id\",\"value\":\"{dataset}\"}}}}]}}&format=json"
        
        try:
            response = requests.get(manifest_url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch manifest: {response.status_code}")
            
            files = response.json()["data"]["hits"]
            
            for file in tqdm(files, desc=f"Downloading {dataset}"):
                file_id = file["file_id"]
                file_name = file["file_name"]
                file_path = dest / file_name
                
                if file_path.exists():
                    continue
                
                download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
                response = requests.get(download_url, stream=True)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    self.logger.error(f"Failed to download {file_name}: {response.status_code}")
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading TCGA dataset {dataset}: {str(e)}")
            raise

class COSMICDownloader(BaseDataSource):
    """Downloader for COSMIC (Catalogue of Somatic Mutations in Cancer) data."""
    
    @property
    def source_name(self) -> str:
        return "cosmic"
    
    def authenticate(self) -> bool:
        """COSMIC requires an API key."""
        api_key = self._get_credential("COSMIC_API_KEY")
        if not api_key:
            self.logger.error("No COSMIC API key found")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available COSMIC datasets."""
        try:
            api_key = self._get_credential("COSMIC_API_KEY")
            if not api_key:
                return []
            
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://cancer.sanger.ac.uk/cosmic/api/v1/datasets", headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch COSMIC datasets: {response.status_code}")
                return []
            
            datasets = []
            for dataset in response.json():
                datasets.append(DatasetInfo(
                    name=dataset["name"],
                    description=dataset.get("description", ""),
                    data_type=DataType.GENOMICS,
                    version=dataset.get("version"),
                    requires_auth=True,
                    url=dataset.get("url")
                ))
            
            return datasets
        except Exception as e:
            self.logger.error(f"Error listing COSMIC datasets: {str(e)}")
            return []
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a COSMIC dataset."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            api_key = self._get_credential("COSMIC_API_KEY")
            if not api_key:
                raise Exception("No COSMIC API key found")
            
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"https://cancer.sanger.ac.uk/cosmic/api/v1/datasets/{dataset}", headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch dataset: {response.status_code}")
            
            dataset_info = response.json()
            download_url = dataset_info["download_url"]
            
            response = requests.get(download_url, headers=headers, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download dataset: {response.status_code}")
            
            file_path = dest / f"{dataset}.gz"
            with open(file_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {dataset}"):
                    f.write(chunk)
            
            # Decompress if needed
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f_in:
                    with open(file_path.with_suffix(''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_path.unlink()
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading COSMIC dataset {dataset}: {str(e)}")
            raise

class ClinVarDownloader(BaseDataSource):
    """Downloader for ClinVar data."""
    
    @property
    def source_name(self) -> str:
        return "clinvar"
    
    def authenticate(self) -> bool:
        """ClinVar is publicly available, no authentication needed."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available ClinVar datasets."""
        return [
            DatasetInfo(
                name="variant_summary",
                description="ClinVar variant summary data",
                data_type=DataType.GENOMICS,
                url="ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/variant_summary.txt.gz"
            ),
            DatasetInfo(
                name="gene_condition_source_id",
                description="ClinVar gene-condition relationships",
                data_type=DataType.GENOMICS,
                url="ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id.txt.gz"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a ClinVar dataset."""
        if not dest:
            dest = self.source_dir
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            ftp_host = "ftp.ncbi.nlm.nih.gov"
            ftp_dir = "/pub/clinvar/"
            
            with ftplib.FTP(ftp_host) as ftp:
                ftp.login()
                ftp.cwd(ftp_dir)
                
                if dataset == "variant_summary":
                    filename = "variant_summary.txt.gz"
                elif dataset == "gene_condition_source_id":
                    filename = "gene_condition_source_id.txt.gz"
                else:
                    raise ValueError(f"Unknown dataset: {dataset}")
                
                file_path = dest / filename
                with open(file_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
                
                # Decompress
                with gzip.open(file_path, 'rb') as f_in:
                    with open(file_path.with_suffix(''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_path.unlink()
                
                return dest
        except Exception as e:
            self.logger.error(f"Error downloading ClinVar dataset {dataset}: {str(e)}")
            raise 