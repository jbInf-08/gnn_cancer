import requests
import ftplib
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo

class CDCDownloader(BaseDataSource):
    """Downloader for CDC (Centers for Disease Control and Prevention) data."""
    
    @property
    def source_name(self) -> str:
        return "cdc"
    
    def authenticate(self) -> bool:
        """CDC data is publicly available."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available CDC datasets."""
        return [
            DatasetInfo(
                name="cancer_incidence",
                description="Cancer Incidence Data",
                data_type=DataType.CLINICAL,
                url="https://www.cdc.gov/cancer/uscs/dataviz/download_data.htm"
            ),
            DatasetInfo(
                name="cancer_mortality",
                description="Cancer Mortality Data",
                data_type=DataType.CLINICAL,
                url="https://www.cdc.gov/cancer/uscs/dataviz/download_data.htm"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download CDC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "cancer_incidence":
                base_url = "https://www.cdc.gov/cancer/uscs/dataviz/download_data.htm"
            elif dataset == "cancer_mortality":
                base_url = "https://www.cdc.gov/cancer/uscs/dataviz/download_data.htm"
            else:
                raise ValueError(f"Unknown CDC dataset: {dataset}")
            
            # CDC data requires manual download
            self.logger.info(f"Please visit {base_url} to download {dataset} data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading CDC dataset {dataset}: {str(e)}")
            raise

class NCBIDownloader(BaseDataSource):
    """Downloader for NCBI (National Center for Biotechnology Information) data."""
    
    @property
    def source_name(self) -> str:
        return "ncbi"
    
    def authenticate(self) -> bool:
        """NCBI data is publicly available."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available NCBI datasets."""
        return [
            DatasetInfo(
                name="pubmed",
                description="PubMed Central Open Access Subset",
                data_type=DataType.METADATA,
                url="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"
            ),
            DatasetInfo(
                name="gene",
                description="Gene Database",
                data_type=DataType.GENOMICS,
                url="ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/"
            ),
            DatasetInfo(
                name="sra",
                description="Sequence Read Archive",
                data_type=DataType.GENOMICS,
                url="https://www.ncbi.nlm.nih.gov/sra"
            ),
            DatasetInfo(
                name="brca1",
                description="BRCA1 Gene Data",
                data_type=DataType.GENOMICS,
                url="ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download NCBI data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "pubmed":
                ftp_host = "ftp.ncbi.nlm.nih.gov"
                ftp_dir = "/pub/pmc/oa_bulk/"
                with ftplib.FTP(ftp_host) as ftp:
                    ftp.login()
                    ftp.cwd(ftp_dir)
                    files = ftp.nlst()
                    for file in tqdm(files, desc="Downloading PubMed files"):
                        if file.endswith(".xml.tar.gz"):
                            file_path = dest / file
                            with open(file_path, 'wb') as f:
                                ftp.retrbinary(f"RETR {file}", f.write)
            
            elif dataset == "gene":
                ftp_host = "ftp.ncbi.nlm.nih.gov"
                ftp_dir = "/gene/DATA/"
                with ftplib.FTP(ftp_host) as ftp:
                    ftp.login()
                    ftp.cwd(ftp_dir)
                    files = ["gene_info.gz", "gene2go.gz", "gene2pubmed.gz"]
                    for file in tqdm(files, desc="Downloading Gene files"):
                        file_path = dest / file
                        with open(file_path, 'wb') as f:
                            ftp.retrbinary(f"RETR {file}", f.write)
            
            elif dataset == "sra":
                # SRA requires the SRA Toolkit
                self.logger.info("Please install the SRA Toolkit to download SRA data.")
                self.logger.info("Visit https://www.ncbi.nlm.nih.gov/sra/docs/toolkitsoft/")
                return dest
            
            elif dataset == "brca1":
                ftp_host = "ftp.ncbi.nlm.nih.gov"
                ftp_dir = "/gene/DATA/"
                with ftplib.FTP(ftp_host) as ftp:
                    ftp.login()
                    ftp.cwd(ftp_dir)
                    files = ["gene_info.gz", "gene2go.gz", "gene2pubmed.gz"]
                    for file in tqdm(files, desc="Downloading BRCA1 Gene files"):
                        file_path = dest / file
                        with open(file_path, 'wb') as f:
                            ftp.retrbinary(f"RETR {file}", f.write)
            
            else:
                raise ValueError(f"Unknown NCBI dataset: {dataset}")
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NCBI dataset {dataset}: {str(e)}")
            raise

class UCIDownloader(BaseDataSource):
    """Downloader for UCI Machine Learning Repository data."""
    
    @property
    def source_name(self) -> str:
        return "uci"
    
    def authenticate(self) -> bool:
        """UCI data is publicly available."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available UCI datasets."""
        return [
            DatasetInfo(
                name="breast_cancer_wisconsin",
                description="Breast Cancer Wisconsin (Diagnostic) Data Set",
                data_type=DataType.CLINICAL,
                url="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"
            ),
            DatasetInfo(
                name="breast_cancer_wisconsin_prognostic",
                description="Breast Cancer Wisconsin (Prognostic) Data Set",
                data_type=DataType.CLINICAL,
                url="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic)"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download UCI dataset."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "breast_cancer_wisconsin":
                base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
                files = ["wdbc.data", "wdbc.names"]
            elif dataset == "breast_cancer_wisconsin_prognostic":
                base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
                files = ["wpbc.data", "wpbc.names"]
            else:
                raise ValueError(f"Unknown UCI dataset: {dataset}")
            
            for file in tqdm(files, desc=f"Downloading {dataset}"):
                response = requests.get(f"{base_url}{file}")
                if response.status_code == 200:
                    file_path = dest / file
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    self.logger.error(f"Failed to download {file}: {response.status_code}")
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading UCI dataset {dataset}: {str(e)}")
            raise

class DDSMDownloader(BaseDataSource):
    """Downloader for DDSM (Digital Database for Screening Mammography) data."""
    
    @property
    def source_name(self) -> str:
        return "ddsm"
    
    def authenticate(self) -> bool:
        """DDSM requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available DDSM datasets."""
        return [
            DatasetInfo(
                name="ddsm",
                description="Digital Database for Screening Mammography",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="http://www.mammoimage.org/databases/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download DDSM data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # DDSM requires manual download after registration
            self.logger.info("Please visit http://www.mammoimage.org/databases/ to register and download DDSM data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading DDSM dataset {dataset}: {str(e)}")
            raise

class INbreastDownloader(BaseDataSource):
    """Downloader for INbreast database."""
    
    @property
    def source_name(self) -> str:
        return "inbreast"
    
    def authenticate(self) -> bool:
        """INbreast requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available INbreast datasets."""
        return [
            DatasetInfo(
                name="inbreast",
                description="INbreast Database",
                data_type=DataType.IMAGING,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.mammoimage.org/databases/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download INbreast data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # INbreast requires manual download after registration
            self.logger.info("Please visit https://www.mammoimage.org/databases/ to register and download INbreast data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading INbreast dataset {dataset}: {str(e)}")
            raise

class NIHDownloader(BaseDataSource):
    """Downloader for NIH (National Institutes of Health) public datasets."""
    @property
    def source_name(self) -> str:
        return "nih"
    def authenticate(self) -> bool:
        """NIH public datasets do not require authentication."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="nih_chest_xray",
                description="NIH Chest X-ray Dataset",
                data_type=DataType.PUBLIC,
                requires_auth=False,
                requires_agreement=False,
                url="https://nihcc.app.box.com/v/ChestXray-NIHCC"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info("Please visit https://nihcc.app.box.com/v/ChestXray-NIHCC to download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NIH dataset {dataset}: {str(e)}")
            raise

class KaggleDownloader(BaseDataSource):
    """Downloader for Kaggle datasets."""
    @property
    def source_name(self) -> str:
        return "kaggle"
    def authenticate(self) -> bool:
        """Kaggle requires API token."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="kaggle_generic",
                description="Kaggle Datasets (generic interface, specify dataset name)",
                data_type=DataType.PUBLIC,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.kaggle.com/datasets"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info("Please use the Kaggle API to download datasets. Example:")
            self.logger.info(f"kaggle datasets download -d {dataset} -p {str(dest)}")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset {dataset}: {str(e)}")
            raise 