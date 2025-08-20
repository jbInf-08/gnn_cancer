import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo

class TCIADownloader(BaseDataSource):
    """Downloader for TCIA (The Cancer Imaging Archive) data."""
    
    @property
    def source_name(self) -> str:
        return "tcia"
    
    def authenticate(self) -> bool:
        """TCIA requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available TCIA datasets."""
        return [
            DatasetInfo(
                name="tcia",
                description="The Cancer Imaging Archive",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.cancerimagingarchive.net/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download TCIA data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # TCIA requires manual download after registration
            self.logger.info("Please visit https://www.cancerimagingarchive.net/ to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading TCIA dataset {dataset}: {str(e)}")
            raise

class MICCAIDownloader(BaseDataSource):
    """Downloader for MICCAI challenge data."""
    
    @property
    def source_name(self) -> str:
        return "miccai"
    
    def authenticate(self) -> bool:
        """MICCAI requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available MICCAI datasets."""
        return [
            DatasetInfo(
                name="miccai",
                description="Medical Image Computing and Computer Assisted Intervention Challenges",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.miccai.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download MICCAI data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # MICCAI requires manual download after registration
            self.logger.info("Please visit https://www.miccai.org/ to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading MICCAI dataset {dataset}: {str(e)}")
            raise

class PathLAIONDownloader(BaseDataSource):
    """Downloader for PathLAION data."""
    
    @property
    def source_name(self) -> str:
        return "pathlaion"
    
    def authenticate(self) -> bool:
        """PathLAION requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available PathLAION datasets."""
        return [
            DatasetInfo(
                name="pathlaion",
                description="PathLAION Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://pathlaion.github.io/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download PathLAION data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # PathLAION requires manual download after registration
            self.logger.info("Please visit https://pathlaion.github.io/ to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading PathLAION dataset {dataset}: {str(e)}")
            raise

class CAMELYONDownloader(BaseDataSource):
    """Downloader for CAMELYON challenge data."""
    
    @property
    def source_name(self) -> str:
        return "camelyon"
    
    def authenticate(self) -> bool:
        """CAMELYON requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available CAMELYON datasets."""
        return [
            DatasetInfo(
                name="camelyon16",
                description="CAMELYON16 Challenge Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://camelyon17.grand-challenge.org/"
            ),
            DatasetInfo(
                name="camelyon17",
                description="CAMELYON17 Challenge Dataset",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://camelyon17.grand-challenge.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download CAMELYON data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # CAMELYON requires manual download after registration
            self.logger.info("Please visit https://camelyon17.grand-challenge.org/ to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading CAMELYON dataset {dataset}: {str(e)}")
            raise

class ISICDownloader(BaseDataSource):
    """Downloader for ISIC Archive data."""
    
    @property
    def source_name(self) -> str:
        return "isic"
    
    def authenticate(self) -> bool:
        """ISIC requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available ISIC datasets."""
        return [
            DatasetInfo(
                name="isic",
                description="International Skin Imaging Collaboration Archive",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.isic-archive.com/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download ISIC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # ISIC requires manual download after registration
            self.logger.info("Please visit https://www.isic-archive.com/ to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading ISIC dataset {dataset}: {str(e)}")
            raise

class HAM10000Downloader(BaseDataSource):
    """Downloader for HAM10000 dataset."""
    
    @property
    def source_name(self) -> str:
        return "ham10000"
    
    def authenticate(self) -> bool:
        """HAM10000 requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available HAM10000 datasets."""
        return [
            DatasetInfo(
                name="ham10000",
                description="HAM10000: A Large Collection of Multi-Source Dermatoscopic Images",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download HAM10000 data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # HAM10000 requires manual download after registration
            self.logger.info("Please visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T to register and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading HAM10000 dataset {dataset}: {str(e)}")
            raise 