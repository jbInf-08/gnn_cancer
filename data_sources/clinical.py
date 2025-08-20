import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo

class MIMICDownloader(BaseDataSource):
    """Downloader for MIMIC-III/IV data."""
    
    @property
    def source_name(self) -> str:
        return "mimic"
    
    def authenticate(self) -> bool:
        """MIMIC requires credentialed access."""
        username = self._get_credential("MIMIC_USERNAME")
        password = self._get_credential("MIMIC_PASSWORD")
        if not username or not password:
            self.logger.error("MIMIC credentials not found")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available MIMIC datasets."""
        return [
            DatasetInfo(
                name="mimiciii",
                description="MIMIC-III Clinical Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/mimiciii/1.4/"
            ),
            DatasetInfo(
                name="mimiciv",
                description="MIMIC-IV Clinical Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/mimiciv/1.0/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download MIMIC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            username = self._get_credential("MIMIC_USERNAME")
            password = self._get_credential("MIMIC_PASSWORD")
            if not username or not password:
                raise Exception("MIMIC credentials not found")
            
            if dataset == "mimiciii":
                base_url = "https://physionet.org/content/mimiciii/1.4/"
            elif dataset == "mimiciv":
                base_url = "https://physionet.org/content/mimiciv/1.0/"
            else:
                raise ValueError(f"Unknown MIMIC version: {dataset}")
            
            # Get file list
            response = requests.get(f"{base_url}files/", auth=(username, password))
            if response.status_code != 200:
                raise Exception(f"Failed to fetch file list: {response.status_code}")
            
            # Download each file
            for file_url in tqdm(response.text.split("\n"), desc=f"Downloading {dataset}"):
                if not file_url.strip():
                    continue
                
                file_name = file_url.split("/")[-1]
                file_path = dest / file_name
                
                if file_path.exists():
                    continue
                
                response = requests.get(f"{base_url}{file_url}", auth=(username, password), stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    self.logger.error(f"Failed to download {file_name}: {response.status_code}")
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading MIMIC dataset {dataset}: {str(e)}")
            raise

class SEERDownloader(BaseDataSource):
    """Downloader for SEER (Surveillance, Epidemiology, and End Results) data."""
    
    @property
    def source_name(self) -> str:
        return "seer"
    
    def authenticate(self) -> bool:
        """SEER requires registration."""
        username = self._get_credential("SEER_USERNAME")
        password = self._get_credential("SEER_PASSWORD")
        if not username or not password:
            self.logger.error("SEER credentials not found")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available SEER datasets."""
        return [
            DatasetInfo(
                name="seer17",
                description="SEER 17 Registries Research Data",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://seer.cancer.gov/data/"
            ),
            DatasetInfo(
                name="seer9",
                description="SEER 9 Registries Research Data",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://seer.cancer.gov/data/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download SEER data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            username = self._get_credential("SEER_USERNAME")
            password = self._get_credential("SEER_PASSWORD")
            if not username or not password:
                raise Exception("SEER credentials not found")
            
            # SEER data requires manual download after registration
            self.logger.info(f"Please visit https://seer.cancer.gov/data/ to download {dataset} data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading SEER dataset {dataset}: {str(e)}")
            raise

class NCDBDownloader(BaseDataSource):
    """Downloader for National Cancer Database (NCDB) data."""
    
    @property
    def source_name(self) -> str:
        return "ncdb"
    
    def authenticate(self) -> bool:
        """NCDB requires registration."""
        username = self._get_credential("NCDB_USERNAME")
        password = self._get_credential("NCDB_PASSWORD")
        if not username or not password:
            self.logger.error("NCDB credentials not found")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available NCDB datasets."""
        return [
            DatasetInfo(
                name="ncdb_puf",
                description="NCDB Participant User File",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.facs.org/quality-programs/cancer/ncdb/puf"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download NCDB data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            username = self._get_credential("NCDB_USERNAME")
            password = self._get_credential("NCDB_PASSWORD")
            if not username or not password:
                raise Exception("NCDB credentials not found")
            
            # NCDB data requires manual download after registration
            self.logger.info("Please visit https://www.facs.org/quality-programs/cancer/ncdb/puf to download NCDB data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NCDB dataset {dataset}: {str(e)}")
            raise 