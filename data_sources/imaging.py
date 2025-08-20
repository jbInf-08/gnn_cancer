import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo

class TCIADownloader(BaseDataSource):
    """Downloader for The Cancer Imaging Archive (TCIA) data."""
    
    @property
    def source_name(self) -> str:
        return "tcia"
    
    def authenticate(self) -> bool:
        """TCIA requires authentication."""
        api_key = self._get_credential("TCIA_API_KEY")
        if not api_key:
            self.logger.error("No TCIA API key found")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available TCIA collections."""
        try:
            api_key = self._get_credential("TCIA_API_KEY")
            if not api_key:
                return []
            
            headers = {"api-key": api_key}
            response = requests.get("https://services.cancerimagingarchive.net/services/v4/TCIA/query/getCollectionValues", headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch TCIA collections: {response.status_code}")
                return []
            
            collections = response.json()
            datasets = []
            
            for collection in collections:
                datasets.append(DatasetInfo(
                    name=collection["Collection"],
                    description=collection.get("Description", ""),
                    data_type=DataType.IMAGING,
                    requires_auth=True,
                    url=f"https://www.cancerimagingarchive.net/collections/{collection['Collection']}/"
                ))
            
            return datasets
        except Exception as e:
            self.logger.error(f"Error listing TCIA collections: {str(e)}")
            return []
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a TCIA collection."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            api_key = self._get_credential("TCIA_API_KEY")
            if not api_key:
                raise Exception("No TCIA API key found")
            
            headers = {"api-key": api_key}
            
            # Get collection details
            collection_url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getCollectionDetails?Collection={dataset}"
            response = requests.get(collection_url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch collection details: {response.status_code}")
            
            # Get series for collection
            series_url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getSeries?Collection={dataset}"
            response = requests.get(series_url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch series: {response.status_code}")
            
            series_list = response.json()
            
            for series in tqdm(series_list, desc=f"Downloading {dataset}"):
                series_uid = series["SeriesInstanceUID"]
                download_url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID={series_uid}"
                
                response = requests.get(download_url, headers=headers, stream=True)
                if response.status_code != 200:
                    self.logger.error(f"Failed to download series {series_uid}: {response.status_code}")
                    continue
                
                series_dir = dest / series_uid
                series_dir.mkdir(exist_ok=True)
                
                # Save DICOM files
                with open(series_dir / f"{series_uid}.dcm", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading TCIA collection {dataset}: {str(e)}")
            raise

class MICCAIDownloader(BaseDataSource):
    """Downloader for MICCAI challenge data."""
    
    @property
    def source_name(self) -> str:
        return "miccai"
    
    def authenticate(self) -> bool:
        """MICCAI challenges may require registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available MICCAI challenges."""
        return [
            DatasetInfo(
                name="brats2020",
                description="Brain Tumor Segmentation Challenge 2020",
                data_type=DataType.IMAGING,
                url="https://www.med.upenn.edu/cbica/brats2020/data.html",
                requires_auth=True,
                requires_agreement=True
            ),
            DatasetInfo(
                name="lits2017",
                description="Liver Tumor Segmentation Challenge 2017",
                data_type=DataType.IMAGING,
                url="https://competitions.codalab.org/competitions/17094",
                requires_auth=True,
                requires_agreement=True
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a MICCAI challenge dataset."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "brats2020":
                # BRATS 2020 requires registration and agreement
                self.logger.info("Please visit https://www.med.upenn.edu/cbica/brats2020/data.html to register and download the data.")
                return dest
            elif dataset == "lits2017":
                # LITS 2017 requires registration and agreement
                self.logger.info("Please visit https://competitions.codalab.org/competitions/17094 to register and download the data.")
                return dest
            else:
                raise ValueError(f"Unknown MICCAI challenge: {dataset}")
        except Exception as e:
            self.logger.error(f"Error downloading MICCAI challenge {dataset}: {str(e)}")
            raise

class ISICDownloader(BaseDataSource):
    """Downloader for ISIC Archive data."""
    
    @property
    def source_name(self) -> str:
        return "isic"
    
    def authenticate(self) -> bool:
        """ISIC Archive requires registration."""
        api_key = self._get_credential("ISIC_API_KEY")
        if not api_key:
            self.logger.warning("No ISIC API key found. Some datasets may be inaccessible.")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available ISIC datasets."""
        try:
            api_key = self._get_credential("ISIC_API_KEY")
            if not api_key:
                return []
            
            headers = {"api-key": api_key}
            response = requests.get("https://isic-archive.com/api/v1/dataset", headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch ISIC datasets: {response.status_code}")
                return []
            
            datasets = []
            for dataset in response.json():
                datasets.append(DatasetInfo(
                    name=dataset["name"],
                    description=dataset.get("description", ""),
                    data_type=DataType.IMAGING,
                    requires_auth=True,
                    url=f"https://isic-archive.com/api/v1/dataset/{dataset['_id']}"
                ))
            
            return datasets
        except Exception as e:
            self.logger.error(f"Error listing ISIC datasets: {str(e)}")
            return []
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download an ISIC dataset."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            api_key = self._get_credential("ISIC_API_KEY")
            if not api_key:
                raise Exception("No ISIC API key found")
            
            headers = {"api-key": api_key}
            
            # Get dataset details
            response = requests.get(f"https://isic-archive.com/api/v1/dataset/{dataset}", headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch dataset details: {response.status_code}")
            
            dataset_info = response.json()
            
            # Download images
            for image_id in tqdm(dataset_info["images"], desc=f"Downloading {dataset}"):
                image_url = f"https://isic-archive.com/api/v1/image/{image_id}/download"
                response = requests.get(image_url, headers=headers, stream=True)
                
                if response.status_code == 200:
                    image_path = dest / f"{image_id}.jpg"
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    self.logger.error(f"Failed to download image {image_id}: {response.status_code}")
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading ISIC dataset {dataset}: {str(e)}")
            raise 