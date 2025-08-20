import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import shutil
from tqdm import tqdm
import pandas as pd
from .base import BaseDataSource, DataType, DatasetInfo
import webbrowser
import json
import logging

class TCGADownloader(BaseDataSource):
    """Downloader for TCGA (The Cancer Genome Atlas) data."""
    
    def __init__(self):
        super().__init__()
        self.token = None
        self.logger = logging.getLogger(__name__)
    
    @property
    def source_name(self) -> str:
        return "tcga"
    
    def authenticate(self) -> bool:
        """TCGA requires registration."""
        # Load token from environment or config
        self.token = os.getenv("GDC_TOKEN")
        if not self.token:
            self.logger.warning("No GDC token found. Some datasets may be inaccessible.")
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available TCGA datasets."""
        return [
            DatasetInfo(
                name="tcga",
                description="The Cancer Genome Atlas",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://portal.gdc.cancer.gov/"
            )
        ]
    
    def get_available_cancer_types(self) -> Dict[str, str]:
        """Get list of available cancer types from TCGA with descriptions."""
        return {
            # Common cancer types
            "BRCA": "Breast Invasive Carcinoma",
            "LUAD": "Lung Adenocarcinoma",
            "LUSC": "Lung Squamous Cell Carcinoma",
            "COAD": "Colon Adenocarcinoma",
            "READ": "Rectum Adenocarcinoma",
            "GBM": "Glioblastoma Multiforme",
            "LGG": "Brain Lower Grade Glioma",
            "OV": "Ovarian Serous Cystadenocarcinoma",
            "UCEC": "Uterine Corpus Endometrial Carcinoma",
            "KIRC": "Kidney Renal Clear Cell Carcinoma",
            "KIRP": "Kidney Renal Papillary Cell Carcinoma",
            "THCA": "Thyroid Carcinoma",
            "PRAD": "Prostate Adenocarcinoma",
            "STAD": "Stomach Adenocarcinoma",
            "SKCM": "Skin Cutaneous Melanoma",
            "BLCA": "Bladder Urothelial Carcinoma",
            "HNSC": "Head and Neck Squamous Cell Carcinoma",
            "LIHC": "Liver Hepatocellular Carcinoma",
            "CESC": "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
            "SARC": "Sarcoma",
            "LAML": "Acute Myeloid Leukemia",
            "PAAD": "Pancreatic Adenocarcinoma",
            "ESCA": "Esophageal Carcinoma",
            "PCPG": "Pheochromocytoma and Paraganglioma",
            "TGCT": "Testicular Germ Cell Tumors",
            "THYM": "Thymoma",
            "ACC": "Adrenocortical Carcinoma",
            "MESO": "Mesothelioma",
            "UVM": "Uveal Melanoma",
            "DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
            "UCS": "Uterine Carcinosarcoma",
            "CHOL": "Cholangiocarcinoma",
            
            # Additional cancer types
            "MELK": "Melanoma",
            "PANCAN": "Pan-Cancer",
            "TCGA": "All TCGA Cancer Types",
            "COADREAD": "Colorectal Adenocarcinoma",
            "GBMLGG": "Glioma",
            "KIPAN": "Kidney Chromophobe",
            "STES": "Stomach and Esophageal Carcinoma",
            "KICH": "Kidney Chromophobe",
            "LGGGBM": "Lower Grade Glioma and Glioblastoma"
        }
    
    def get_available_data_types(self) -> Dict[str, str]:
        """Get list of available data types with descriptions."""
        return {
            "Simple Nucleotide Variation": "Mutation data",
            "Gene Expression Quantification": "RNA-seq expression data",
            "Copy Number Variation": "CNV data",
            "DNA Methylation": "Methylation data",
            "miRNA Expression Quantification": "miRNA expression data",
            "Protein Expression Quantification": "Protein expression data",
            "Clinical": "Clinical data",
            "Pathology": "Pathology data",
            "Biospecimen": "Biospecimen data",
            "Slide Image": "Pathology slide images",
            "Raw Sequencing Data": "Raw sequencing data",
            "Aligned Reads": "Aligned sequencing reads",
            "Gene Expression": "Gene expression data",
            "Isoform Expression Quantification": "Isoform expression data",
            "Exon Expression Quantification": "Exon expression data",
            "Exon Junction Quantification": "Exon junction data",
            "Transcriptome Profiling": "Transcriptome data",
            "Somatic Mutation": "Somatic mutation data",
            "Germline Mutation": "Germline mutation data",
            "Structural Variation": "Structural variation data"
        }
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download TCGA data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get user input for cancer type and data type
            print("\nAvailable cancer types:")
            cancer_types = self.get_available_cancer_types()
            for i, (code, name) in enumerate(cancer_types.items(), 1):
                print(f"{i}. {code} - {name}")
            
            while True:
                try:
                    choice = int(input("\nSelect cancer type (number): "))
                    if 1 <= choice <= len(cancer_types):
                        cancer_type = list(cancer_types.keys())[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            print("\nAvailable data types:")
            data_types = self.get_available_data_types()
            for i, (code, name) in enumerate(data_types.items(), 1):
                print(f"{i}. {code} - {name}")
            
            while True:
                try:
                    choice = int(input("\nSelect data type (number): "))
                    if 1 <= choice <= len(data_types):
                        data_type = list(data_types.keys())[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            # Construct the GDC API query
            filters = {
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
                    }
                ]
            }
            
            # Get file IDs
            url = "https://api.gdc.cancer.gov/files"
            params = {
                "filters": json.dumps(filters),
                "fields": "file_id,file_name,file_size,data_type,data_format",
                "size": "1000"
            }
            
            headers = {}
            if self.token:
                headers["X-Auth-Token"] = self.token
            
            self.logger.info(f"Querying TCGA for {cancer_type} {data_type} data...")
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 401:
                self.logger.error("Authentication required. Please set up your TCGA token first.")
                print("\nTo get your TCGA token:")
                print("1. Visit https://portal.gdc.cancer.gov/")
                print("2. Log in to your account")
                print("3. Go to your profile")
                print("4. Copy your token")
                print("5. Run the setup script: python setup_credentials.py")
                return dest
            
            response.raise_for_status()
            data = response.json()
            
            if not data["data"]["hits"]:
                self.logger.warning(f"No {data_type} data found for {cancer_type}")
                print("\nAvailable data types for this cancer type:")
                # Query available data types
                type_url = "https://api.gdc.cancer.gov/files"
                type_params = {
                    "filters": json.dumps({
                        "op": "=",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": f"TCGA-{cancer_type}"
                        }
                    }),
                    "fields": "data_type",
                    "size": "0",
                    "facets": "data_type"
                }
                type_response = requests.get(type_url, params=type_params, headers=headers)
                type_response.raise_for_status()
                type_data = type_response.json()
                
                if "facets" in type_data and "data_type" in type_data["facets"]:
                    for dt in type_data["facets"]["data_type"]:
                        print(f"- {dt['key']}: {dt['doc_count']} files")
                return dest
            
            # Download files
            for file in data["data"]["hits"]:
                file_id = file["file_id"]
                file_name = file["file_name"]
                file_path = dest / file_name
                
                if file_path.exists():
                    self.logger.info(f"File {file_name} already exists, skipping...")
                    continue
                
                self.logger.info(f"Downloading {file_name}...")
                download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
                response = requests.get(download_url, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"Successfully downloaded {file_name}")
            
            return dest
            
        except Exception as e:
            self.logger.error(f"Error downloading TCGA dataset {dataset}: {str(e)}")
            raise

class ICGCDownloader(BaseDataSource):
    """Downloader for ICGC (International Cancer Genome Consortium) data."""
    
    @property
    def source_name(self) -> str:
        return "icgc"
    
    def authenticate(self) -> bool:
        """ICGC requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available ICGC datasets."""
        return [
            DatasetInfo(
                name="icgc",
                description="International Cancer Genome Consortium",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://dcc.icgc.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download ICGC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://dcc.icgc.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading ICGC dataset {dataset}: {str(e)}")
            raise

class EGADownloader(BaseDataSource):
    """Downloader for EGA (European Genome-phenome Archive) data."""
    
    @property
    def source_name(self) -> str:
        return "ega"
    
    def authenticate(self) -> bool:
        """EGA requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available EGA datasets."""
        return [
            DatasetInfo(
                name="ega",
                description="European Genome-phenome Archive",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://ega-archive.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download EGA data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://ega-archive.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading EGA dataset {dataset}: {str(e)}")
            raise

class COSMICDownloader(BaseDataSource):
    """Downloader for COSMIC (Catalogue of Somatic Mutations in Cancer) data."""
    
    @property
    def source_name(self) -> str:
        return "cosmic"
    
    def authenticate(self) -> bool:
        """COSMIC requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available COSMIC datasets."""
        return [
            DatasetInfo(
                name="cosmic",
                description="Catalogue of Somatic Mutations in Cancer",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://cancer.sanger.ac.uk/cosmic"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download COSMIC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://cancer.sanger.ac.uk/cosmic"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
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
        """ClinVar requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available ClinVar datasets."""
        return [
            DatasetInfo(
                name="clinvar",
                description="ClinVar Database",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.ncbi.nlm.nih.gov/clinvar/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download ClinVar data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://www.ncbi.nlm.nih.gov/clinvar/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading ClinVar dataset {dataset}: {str(e)}")
            raise

class OncoKBDownloader(BaseDataSource):
    """Downloader for OncoKB (Precision Oncology Knowledge Base) data."""
    
    @property
    def source_name(self) -> str:
        return "oncokb"
    
    def authenticate(self) -> bool:
        """OncoKB requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available OncoKB datasets."""
        return [
            DatasetInfo(
                name="oncokb",
                description="Precision Oncology Knowledge Base",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.oncokb.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download OncoKB data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://www.oncokb.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading OncoKB dataset {dataset}: {str(e)}")
            raise

class CBioPortalDownloader(BaseDataSource):
    """Downloader for cBioPortal data."""
    
    @property
    def source_name(self) -> str:
        return "cbioportal"
    
    def authenticate(self) -> bool:
        """cBioPortal requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available cBioPortal datasets."""
        return [
            DatasetInfo(
                name="cbioportal",
                description="cBioPortal for Cancer Genomics",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.cbioportal.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download cBioPortal data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://www.cbioportal.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading cBioPortal dataset {dataset}: {str(e)}")
            raise

class CCLEDownloader(BaseDataSource):
    """Downloader for CCLE (Cancer Cell Line Encyclopedia) data."""
    
    @property
    def source_name(self) -> str:
        return "ccle"
    
    def authenticate(self) -> bool:
        """CCLE requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available CCLE datasets."""
        return [
            DatasetInfo(
                name="ccle",
                description="Cancer Cell Line Encyclopedia",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://depmap.org/portal/ccle/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download CCLE data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://depmap.org/portal/ccle/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading CCLE dataset {dataset}: {str(e)}")
            raise

class GDSCDownloader(BaseDataSource):
    """Downloader for GDSC (Genomics of Drug Sensitivity in Cancer) data."""
    
    @property
    def source_name(self) -> str:
        return "gdsc"
    
    def authenticate(self) -> bool:
        """GDSC requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available GDSC datasets."""
        return [
            DatasetInfo(
                name="gdsc",
                description="Genomics of Drug Sensitivity in Cancer",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.cancerrxgene.org/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download GDSC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://www.cancerrxgene.org/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading GDSC dataset {dataset}: {str(e)}")
            raise

class NCI60Downloader(BaseDataSource):
    """Downloader for NCI-60 data."""
    
    @property
    def source_name(self) -> str:
        return "nci60"
    
    def authenticate(self) -> bool:
        """NCI-60 requires registration."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available NCI-60 datasets."""
        return [
            DatasetInfo(
                name="nci60",
                description="NCI-60 Human Tumor Cell Lines Screen",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://dtp.cancer.gov/discovery_development/nci-60/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download NCI-60 data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            url = "https://dtp.cancer.gov/discovery_development/nci-60/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NCI-60 dataset {dataset}: {str(e)}")
            raise

class FireCloudTerraDownloader(BaseDataSource):
    """Downloader for Broad Institute's FireCloud/Terra platform."""
    @property
    def source_name(self) -> str:
        return "firecloud_terra"
    def authenticate(self) -> bool:
        """FireCloud/Terra requires registration and API key."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="firecloud_terra",
                description="Broad Institute's FireCloud/Terra Platform",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://terra.bio/"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info("Please visit https://terra.bio/ to register and download the data using the platform's interface or API.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading FireCloud/Terra dataset {dataset}: {str(e)}")
            raise

class GoogleCloudHealthcareDownloader(BaseDataSource):
    """Downloader for Google Cloud Healthcare API datasets."""
    @property
    def source_name(self) -> str:
        return "google_cloud_healthcare"
    def authenticate(self) -> bool:
        """Google Cloud Healthcare API requires authentication and project setup."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="google_cloud_healthcare",
                description="Google Cloud Healthcare API Datasets",
                data_type=DataType.GENOMICS,
                requires_auth=True,
                requires_agreement=True,
                url="https://cloud.google.com/healthcare"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info("Please visit https://cloud.google.com/healthcare to set up access and download the data using the API.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading Google Cloud Healthcare dataset {dataset}: {str(e)}")
            raise 