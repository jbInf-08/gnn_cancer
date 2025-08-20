from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from enum import Enum
import json

class DataType(Enum):
    """Types of data sources."""
    GENOMICS = "genomics"
    IMAGING = "imaging"
    CLINICAL = "clinical"
    PUBLIC = "public"
    METADATA = "metadata"

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    data_type: DataType
    requires_auth: bool = False
    requires_agreement: bool = False
    url: Optional[str] = None

class BaseDataSource(ABC):
    """Base class for all data sources."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """Initialize the data source.
        
        Args:
            data_dir: Directory to store downloaded data. If None, uses 'data/raw/{source_name}'.
        """
        self.logger = logging.getLogger(f"data_downloader.{self.source_name}")
        
        if data_dir is None:
            data_dir = Path("data/raw") / self.source_name
        self.source_dir = Path(data_dir)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        
        # Load credentials if they exist
        self.credentials = self._load_credentials()
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source."""
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the data source.
        
        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def list_available(self) -> List[DatasetInfo]:
        """List all available datasets from this source.
        
        Returns:
            List[DatasetInfo]: List of available datasets.
        """
        pass
    
    @abstractmethod
    def download(self, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a dataset.
        
        Args:
            dataset: Name of the dataset to download.
            dest: Destination directory. If None, uses source_dir/dataset.
            
        Returns:
            Path: Path to the downloaded data.
            
        Raises:
            ValueError: If dataset is not available.
            Exception: If download fails.
        """
        pass
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from environment variables or config file.
        
        Returns:
            Dict[str, str]: Dictionary of credentials.
        """
        # First try environment variables
        prefix = f"{self.source_name.upper()}_"
        credentials = {}
        for key in ["API_KEY", "USERNAME", "PASSWORD", "TOKEN"]:
            env_key = prefix + key
            if env_key in os.environ:
                credentials[key.lower()] = os.environ[env_key]
        
        # Then try config file
        config_file = Path("config") / f"{self.source_name}_credentials.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_creds = json.load(f)
                credentials.update(file_creds)
            except Exception as e:
                self.logger.warning(f"Error loading credentials from {config_file}: {str(e)}")
        
        return credentials
    
    def _save_credentials(self) -> None:
        """Save credentials to config file."""
        if not self.credentials:
            return
        
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"{self.source_name}_credentials.json"
        try:
            with open(config_file, "w") as f:
                json.dump(self.credentials, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving credentials to {config_file}: {str(e)}")
    
    def _check_credentials(self, required_keys: List[str]) -> bool:
        """Check if required credentials are present.
        
        Args:
            required_keys: List of required credential keys.
            
        Returns:
            bool: True if all required credentials are present, False otherwise.
        """
        missing = [key for key in required_keys if key not in self.credentials]
        if missing:
            self.logger.error(f"Missing required credentials: {', '.join(missing)}")
            return False
        return True
    
    def _create_dest_dir(self, dest: Optional[Path] = None) -> Path:
        """Create destination directory.
        
        Args:
            dest: Destination directory. If None, uses source_dir.
            
        Returns:
            Path: Path to the created directory.
        """
        if dest is None:
            dest = self.source_dir
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

class DataDownloader:
    """Main class to manage all data sources."""
    
    def __init__(self, data_dir: Union[str, Path], credentials: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.credentials = credentials or {}
        self.sources: Dict[str, BaseDataSource] = {}
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def register_source(self, source: BaseDataSource):
        """Register a new data source."""
        self.sources[source.source_name] = source
    
    def get_source(self, source_name: str) -> BaseDataSource:
        """Get a registered data source."""
        if source_name not in self.sources:
            raise ValueError(f"Source {source_name} not registered")
        return self.sources[source_name]
    
    def list_all_datasets(self) -> Dict[str, List[DatasetInfo]]:
        """List all available datasets from all sources."""
        datasets = {}
        for source_name, source in self.sources.items():
            try:
                datasets[source_name] = source.list_available()
            except Exception as e:
                self.logger.error(f"Error listing datasets from {source_name}: {str(e)}")
                datasets[source_name] = []
        return datasets
    
    def download_dataset(self, source_name: str, dataset: str, dest: Optional[Path] = None) -> Path:
        """Download a dataset from a specific source."""
        source = self.get_source(source_name)
        return source.download(dataset, dest)
    
    def authenticate_all(self) -> Dict[str, bool]:
        """Authenticate with all sources that require authentication."""
        results = {}
        for source_name, source in self.sources.items():
            try:
                results[source_name] = source.authenticate()
            except Exception as e:
                self.logger.error(f"Error authenticating with {source_name}: {str(e)}")
                results[source_name] = False
        return results 