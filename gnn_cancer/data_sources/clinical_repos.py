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

class MIMICDownloader(BaseDataSource):
    """Downloader for MIMIC (Medical Information Mart for Intensive Care) data."""
    
    @property
    def source_name(self) -> str:
        return "mimic"
    
    def authenticate(self) -> bool:
        """MIMIC requires registration and completion of CITI training."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available MIMIC datasets."""
        return [
            DatasetInfo(
                name="mimic3",
                description="MIMIC-III Clinical Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/mimiciii/1.4/"
            ),
            DatasetInfo(
                name="mimic4",
                description="MIMIC-IV Clinical Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/mimiciv/2.0/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download MIMIC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if dataset == "mimic3":
                url = "https://physionet.org/content/mimiciii/1.4/"
            elif dataset == "mimic4":
                url = "https://physionet.org/content/mimiciv/2.0/"
            else:
                raise ValueError(f"Unknown MIMIC dataset: {dataset}")
            
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading MIMIC dataset {dataset}: {str(e)}")
            raise

class eICUDownloader(BaseDataSource):
    """Downloader for eICU Collaborative Research Database."""
    
    @property
    def source_name(self) -> str:
        return "eicu"
    
    def authenticate(self) -> bool:
        """eICU requires registration and completion of CITI training."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available eICU datasets."""
        return [
            DatasetInfo(
                name="eicu",
                description="eICU Collaborative Research Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://eicu-crd.mit.edu/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download eICU data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # eICU requires manual download after registration and CITI training
            self.logger.info("Please visit https://eicu-crd.mit.edu/ to register, complete CITI training, and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading eICU dataset {dataset}: {str(e)}")
            raise

class HiRIDDownloader(BaseDataSource):
    """Downloader for HiRID (High Time Resolution ICU Dataset)."""
    
    @property
    def source_name(self) -> str:
        return "hirid"
    
    def authenticate(self) -> bool:
        """HiRID requires registration and completion of CITI training."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available HiRID datasets."""
        return [
            DatasetInfo(
                name="hirid",
                description="High Time Resolution ICU Dataset",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/hirid/1.1.1/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download HiRID data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # HiRID requires manual download after registration and CITI training
            self.logger.info("Please visit https://physionet.org/content/hirid/1.1.1/ to register, complete CITI training, and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading HiRID dataset {dataset}: {str(e)}")
            raise

class AUMCDownloader(BaseDataSource):
    """Downloader for Amsterdam UMC Database."""
    
    @property
    def source_name(self) -> str:
        return "aumc"
    
    def authenticate(self) -> bool:
        """AUMC requires registration and completion of CITI training."""
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        """List available AUMC datasets."""
        return [
            DatasetInfo(
                name="aumc",
                description="Amsterdam UMC Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://amsterdammedicaldatascience.nl/"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download AUMC data."""
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # AUMC requires manual download after registration and CITI training
            self.logger.info("Please visit https://amsterdammedicaldatascience.nl/ to register, complete CITI training, and download the data.")
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading AUMC dataset {dataset}: {str(e)}")
            raise

class SEERDownloader(BaseDataSource):
    """Downloader for SEER (Surveillance, Epidemiology, and End Results) data."""
    @property
    def source_name(self) -> str:
        return "seer"
    def authenticate(self) -> bool:
        """SEER requires registration and data use agreement."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="seer",
                description="Surveillance, Epidemiology, and End Results Program",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://seer.cancer.gov/data/"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            url = "https://seer.cancer.gov/data/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading SEER dataset {dataset}: {str(e)}")
            raise

class NCDBDownloader(BaseDataSource):
    """Downloader for NCDB (National Cancer Database) data."""
    @property
    def source_name(self) -> str:
        return "ncdb"
    def authenticate(self) -> bool:
        """NCDB requires registration and data use agreement."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="ncdb",
                description="National Cancer Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.facs.org/quality-programs/cancer/ncdb/"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            url = "https://www.facs.org/quality-programs/cancer/ncdb/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading NCDB dataset {dataset}: {str(e)}")
            raise

class MIMICIIDownloader(BaseDataSource):
    """Downloader for MIMIC-II (Medical Information Mart for Intensive Care II) data."""
    @property
    def source_name(self) -> str:
        return "mimic2"
    def authenticate(self) -> bool:
        """MIMIC-II requires registration and CITI training."""
        return True
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="mimic2",
                description="MIMIC-II Clinical Database",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://physionet.org/content/mimic2demo/1.0/"
            )
        ]
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        try:
            url = "https://physionet.org/content/mimic2demo/1.0/"
            self.logger.info(f"Opening {url} in your browser for manual download.")
            webbrowser.open(url)
            self.logger.info("After downloading, place the files in the following directory:")
            self.logger.info(str(dest))
            return dest
        except Exception as e:
            self.logger.error(f"Error downloading MIMIC-II dataset {dataset}: {str(e)}")
            raise

class COSMICDownloader(BaseDataSource):
    @property
    def source_name(self) -> str:
        return "cosmic"
    
    def authenticate(self) -> bool:
        # No authentication needed for public API
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="cosmic_mutations",
                description="COSMIC Mutations Database (via NLM Clinical Tables API)",
                data_type=DataType.CLINICAL,
                requires_auth=False,
                requires_agreement=False,
                url="https://clinicaltables.nlm.nih.gov/api/cosmic/v4/search"
            )
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # List of cancer-related genes to query
            genes = [
                "BRCA1", "BRCA2", "TP53", "KRAS", "BRAF", "EGFR", "PIK3CA",
                "PTEN", "RB1", "CDKN2A", "MYC", "ALK", "ROS1", "MET", "HER2"
            ]
            
            all_mutations = []
            for gene in genes:
                url = f"https://clinicaltables.nlm.nih.gov/api/cosmic/v4/search"
                params = {
                    "terms": gene,
                    "df": "MutationID,GeneName,MutationCDS,MutationAA",
                    "maxList": 500
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract mutations from response
                mutations = data[1]  # The actual mutations are in the second element
                all_mutations.extend(mutations)
                
                self.logger.info(f"Downloaded {len(mutations)} mutations for {gene}")
            
            # Save to file
            output_file = dest / "cosmic_mutations.json"
            with open(output_file, "w") as f:
                json.dump(all_mutations, f, indent=2)
            
            self.logger.info(f"Saved {len(all_mutations)} total mutations to {output_file}")
            return dest
            
        except Exception as e:
            self.logger.error(f"Error downloading COSMIC data: {str(e)}")
            raise

class TCGADownloader(BaseDataSource):
    def __init__(self):
        super().__init__()
        self.token = None
        self.authenticate()
    
    @property
    def source_name(self) -> str:
        return "tcga"
    
    def authenticate(self) -> bool:
        # Load token from config
        config_file = Path("config/api_keys.json")
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                self.token = config.get("TCGA_TOKEN", "")
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="tcga",
                description="The Cancer Genome Atlas",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://portal.gdc.cancer.gov/"
            )
        ]
    
    def get_available_cancer_types(self) -> List[str]:
        """Get list of available cancer types from TCGA."""
        return [
            "BRCA", "LUAD", "LUSC", "COAD", "READ", "GBM", "LGG", "OV", "UCEC",
            "KIRC", "KIRP", "THCA", "PRAD", "STAD", "SKCM", "BLCA", "HNSC",
            "LIHC", "CESC", "SARC", "LAML", "PAAD", "ESCA", "PCPG", "TGCT",
            "THYM", "ACC", "MESO", "UVM", "DLBC", "UCS", "CHOL"
        ]
    
    def get_available_data_types(self) -> List[str]:
        """Get list of available data types."""
        return [
            "Simple Nucleotide Variation",  # mutation
            "Gene Expression Quantification",  # expression
            "Copy Number Variation",  # cnv
            "DNA Methylation",  # methylation
            "miRNA Expression Quantification",  # mirna
            "Protein Expression Quantification",  # protein
            "Clinical",  # clinical
            "Pathology"  # pathology
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get user input for cancer type and data type
            print("\nAvailable cancer types:")
            cancer_types = self.get_available_cancer_types()
            for i, ct in enumerate(cancer_types, 1):
                print(f"{i}. {ct}")
            
            while True:
                try:
                    choice = int(input("\nSelect cancer type (number): "))
                    if 1 <= choice <= len(cancer_types):
                        cancer_type = cancer_types[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            print("\nAvailable data types:")
            data_types = self.get_available_data_types()
            for i, dt in enumerate(data_types, 1):
                print(f"{i}. {dt}")
            
            while True:
                try:
                    choice = int(input("\nSelect data type (number): "))
                    if 1 <= choice <= len(data_types):
                        data_type = data_types[choice - 1]
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
            
            # Download each file
            for hit in data["data"]["hits"]:
                file_id = hit["id"]
                file_name = hit["file_name"]
                file_size = hit["file_size"]
                
                self.logger.info(f"Downloading {file_name} ({file_size/1024/1024:.1f} MB)...")
                
                # Download file
                download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
                response = requests.get(download_url, headers=headers, stream=True)
                response.raise_for_status()
                
                output_file = dest / file_name
                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                self.logger.info(f"Downloaded {file_name}")
            
            self.logger.info(f"Downloaded {len(data['data']['hits'])} files to {dest}")
            return dest
            
        except Exception as e:
            self.logger.error(f"Error downloading TCGA data: {str(e)}")
            raise

class KaggleDownloader(BaseDataSource):
    def __init__(self):
        super().__init__()
        self.kaggle_dir = Path.home() / ".kaggle"
        self.kaggle_json = self.kaggle_dir / "kaggle.json"
    
    @property
    def source_name(self) -> str:
        return "kaggle"
    
    def authenticate(self) -> bool:
        if not self.kaggle_json.exists():
            self.logger.error("Kaggle API credentials not found.")
            print("\nTo set up Kaggle API:")
            print("1. Visit https://www.kaggle.com/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print(f"4. Place the downloaded kaggle.json file in: {self.kaggle_dir}")
            return False
        return True
    
    def list_available(self) -> List[DatasetInfo]:
        return [
            DatasetInfo(
                name="cancer_datasets",
                description="Various cancer-related datasets from Kaggle",
                data_type=DataType.CLINICAL,
                requires_auth=True,
                requires_agreement=True,
                url="https://www.kaggle.com/datasets"
            )
        ]
    
    def get_available_datasets(self) -> List[Dict[str, str]]:
        """Get list of available cancer-related datasets."""
        return [
            {
                "name": "breast-cancer-wisconsin",
                "owner": "uciml",
                "description": "Breast Cancer Wisconsin (Diagnostic) Data Set"
            },
            {
                "name": "cervical-cancer-risk-classification",
                "owner": "fedesoriano",
                "description": "Cervical Cancer Risk Classification"
            },
            {
                "name": "lung-cancer",
                "owner": "aryashah2k",
                "description": "Lung Cancer Dataset"
            },
            {
                "name": "prostate-cancer",
                "owner": "jessemostipak",
                "description": "Prostate Cancer Dataset"
            }
        ]
    
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        if not dest:
            dest = self.source_dir / dataset
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            if not self.authenticate():
                return dest
            
            # Get user input for dataset
            print("\nAvailable datasets:")
            datasets = self.get_available_datasets()
            for i, ds in enumerate(datasets, 1):
                print(f"{i}. {ds['description']} ({ds['owner']}/{ds['name']})")
            
            while True:
                try:
                    choice = int(input("\nSelect dataset (number): "))
                    if 1 <= choice <= len(datasets):
                        selected = datasets[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            # Download dataset using kaggle API
            dataset_path = f"{selected['owner']}/{selected['name']}"
            self.logger.info(f"Downloading {dataset_path}...")
            
            # Use subprocess to run kaggle command
            import subprocess
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_path, "-p", str(dest)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Error downloading dataset: {result.stderr}")
                return dest
            
            self.logger.info(f"Downloaded {dataset_path} to {dest}")
            return dest
            
        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset: {str(e)}")
            raise 