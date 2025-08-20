import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo
import webbrowser

class LIDCIDRIDownloader(BaseDataSource):
    """Downloader for LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative) data."""
    
    @property
    def source_name(self) -> str:
        return "lidc_idri"
    
    def authenticate(self) -> bool:
        """LIDC-IDRI requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available LIDC-IDRI datasets."""
        return [
            DatasetInfo(
                name="lidc_idri",
                description="Lung Image Database Consortium and Image Database Resource Initiative",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download LIDC-IDRI data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading LIDC-IDRI dataset {dataset}: {str(e)}")
            raise

class NSCLCDownloader(BaseDataSource):
    """Downloader for NSCLC (Non-Small Cell Lung Cancer) Radiogenomics data."""
    
    @property
    def source_name(self) -> str:
        return "nsclc"
    
    def authenticate(self) -> bool:
        """NSCLC requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available NSCLC datasets."""
        return [
            DatasetInfo(
                name="nsclc_radiogenomics",
                description="NSCLC Radiogenomics Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download NSCLC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NSCLC dataset {dataset}: {str(e)}")
            raise

class Luna16Downloader(BaseDataSource):
    """Downloader for LUNA16 (Lung Nodule Analysis) challenge data."""
    
    @property
    def source_name(self) -> str:
        return "luna16"
    
    def authenticate(self) -> bool:
        """LUNA16 requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available LUNA16 datasets."""
        return [
            DatasetInfo(
                name="luna16",
                description="LUNA16 Challenge Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://luna16.grand-challenge.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download LUNA16 data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://luna16.grand-challenge.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading LUNA16 dataset {dataset}: {str(e)}")
            raise

class BraTSDownloader(BaseDataSource):
    """Downloader for BraTS (Brain Tumor Segmentation) challenge data."""
    
    @property
    def source_name(self) -> str:
        return "brats"
    
    def authenticate(self) -> bool:
        """BraTS requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available BraTS datasets."""
        return [
            DatasetInfo(
                name="brats2020",
                description="BraTS 2020 Challenge Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.med.upenn.edu/cbica/brats2020/data.html"
            ),
            DatasetInfo(
                name="brats2021",
                description="BraTS 2021 Challenge Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.med.upenn.edu/cbica/brats2021/data.html"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download BraTS data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "brats2020":
                url = "https://www.med.upenn.edu/cbica/brats2020/data.html"
            elif dataset == "brats2021":
                url = "https://www.med.upenn.edu/cbica/brats2021/data.html"
            else:
                raise ValueError(f"Unknown BraTS dataset: {dataset}")
            
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading BraTS dataset {dataset}: {str(e)}")
            raise

class REMBRANDTDownloader(BaseDataSource):
    """Downloader for REMBRANDT (Repository for Molecular Brain Neoplasia Data) data."""
    
    @property
    def source_name(self) -> str:
        return "rembrandt"
    
    def authenticate(self) -> bool:
        """REMBRANDT requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available REMBRANDT datasets."""
        return [
            DatasetInfo(
                name="rembrandt",
                description="REMBRANDT Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download REMBRANDT data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading REMBRANDT dataset {dataset}: {str(e)}")
            raise 