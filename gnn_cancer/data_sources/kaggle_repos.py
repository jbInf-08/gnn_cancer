from pathlib import Path
from typing import List, Dict, Optional
import json
import kaggle
from .base import BaseDataSource, DataType, DatasetInfo

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
            
            # Use kaggle API to download
            kaggle.api.dataset_download_files(
                dataset_path,
                path=str(dest),
                unzip=True
            )
            
            self.logger.info(f"Downloaded {dataset_path} to {dest}")
            return dest
            
        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset: {str(e)}")
            raise 