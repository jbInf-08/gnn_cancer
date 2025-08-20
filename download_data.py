import os
from pathlib import Path
import logging
from typing import Dict, Optional, List
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

from data_sources.base import DataDownloader, BaseDataSource, DataType, DatasetInfo
from data_sources.public_repos import (
    CDCDownloader,
    NCBIDownloader,
    UCIDownloader,
    DDSMDownloader,
    INbreastDownloader,
    NIHDownloader,
    KaggleDownloader
)
from data_sources.imaging_challenges import (
    LIDCIDRIDownloader,
    NSCLCDownloader,
    Luna16Downloader,
    BraTSDownloader,
    REMBRANDTDownloader
)
from data_sources.clinical_repos import (
    MIMICDownloader,
    eICUDownloader,
    HiRIDDownloader,
    AUMCDownloader,
    SEERDownloader,
    NCDBDownloader,
    MIMICIIDownloader
)
from data_sources.genomics_repos import (
    TCGADownloader,
    ICGCDownloader,
    EGADownloader,
    COSMICDownloader,
    ClinVarDownloader,
    OncoKBDownloader,
    CBioPortalDownloader,
    CCLEDownloader,
    GDSCDownloader,
    NCI60Downloader,
    FireCloudTerraDownloader,
    GoogleCloudHealthcareDownloader
)

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("data_downloader")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_credentials() -> Dict[str, str]:
    """Load credentials from environment variables."""
    load_dotenv()  # Load from .env file
    
    credentials = {}
    credential_keys = [
        # TCGA
        "GDC_TOKEN",
        # COSMIC
        "COSMIC_API_KEY",
        # TCIA
        "TCIA_API_KEY",
        # ISIC
        "ISIC_API_KEY",
        # MIMIC
        "MIMIC_USERNAME",
        "MIMIC_PASSWORD",
        # SEER
        "SEER_USERNAME",
        "SEER_PASSWORD",
        # NCDB
        "NCDB_USERNAME",
        "NCDB_PASSWORD"
    ]
    
    for key in credential_keys:
        value = os.getenv(key)
        if value:
            credentials[key] = value
    
    return credentials

def get_all_downloaders() -> List[BaseDataSource]:
    """Get all available data downloaders."""
    return [
        # Public repositories
        CDCDownloader(),
        NCBIDownloader(),
        UCIDownloader(),
        DDSMDownloader(),
        INbreastDownloader(),
        NIHDownloader(),
        KaggleDownloader(),
        
        # Imaging challenges
        LIDCIDRIDownloader(),
        NSCLCDownloader(),
        Luna16Downloader(),
        BraTSDownloader(),
        REMBRANDTDownloader(),
        
        # Clinical repositories
        MIMICDownloader(),
        eICUDownloader(),
        HiRIDDownloader(),
        AUMCDownloader(),
        SEERDownloader(),
        NCDBDownloader(),
        MIMICIIDownloader(),
        
        # Genomics repositories
        TCGADownloader(),
        ICGCDownloader(),
        EGADownloader(),
        COSMICDownloader(),
        ClinVarDownloader(),
        OncoKBDownloader(),
        CBioPortalDownloader(),
        CCLEDownloader(),
        GDSCDownloader(),
        NCI60Downloader(),
        FireCloudTerraDownloader(),
        GoogleCloudHealthcareDownloader()
    ]

def list_available_datasets(data_type: Optional[DataType] = None) -> None:
    """List all available datasets, optionally filtered by data type."""
    logger = setup_logging()
    downloaders = get_all_downloaders()
    
    logger.info("Available datasets:")
    for downloader in downloaders:
        datasets = downloader.list_available()
        for dataset in datasets:
            if data_type is None or dataset.data_type == data_type:
                logger.info(f"\nSource: {downloader.source_name}")
                logger.info(f"Dataset: {dataset.name}")
                logger.info(f"Description: {dataset.description}")
                logger.info(f"Type: {dataset.data_type}")
                logger.info(f"URL: {dataset.url}")
                if dataset.requires_auth:
                    logger.info("Requires authentication: Yes")
                if dataset.requires_agreement:
                    logger.info("Requires data use agreement: Yes")
                logger.info("-" * 80)

def download_dataset(source_name: str, dataset_name: str, dest_dir: Optional[Path] = None) -> None:
    """Download a specific dataset from a specific source."""
    logger = setup_logging()
    downloaders = get_all_downloaders()
    
    # Find the appropriate downloader
    downloader = next((d for d in downloaders if d.source_name == source_name), None)
    if not downloader:
        logger.error(f"Unknown source: {source_name}")
        return
    
    # Check if dataset is available
    available_datasets = downloader.list_available()
    if not any(d.name == dataset_name for d in available_datasets):
        logger.error(f"Dataset {dataset_name} not available from source {source_name}")
        return
    
    # Authenticate if required
    if not downloader.authenticate():
        logger.error(f"Authentication failed for source {source_name}")
        return
    
    # Download the dataset
    try:
        dest_path = downloader.download(dataset_name, dest_dir)
        logger.info(f"Successfully downloaded {dataset_name} to {dest_path}")
    except Exception as e:
        logger.error(f"Error downloading {dataset_name} from {source_name}: {str(e)}")

def auto_download_all(dest_dir: Optional[Path] = None):
    """Attempt to download all datasets that can be fetched programmatically."""
    logger = setup_logging()
    downloaders = get_all_downloaders()
    for downloader in downloaders:
        try:
            datasets = downloader.list_available()
            for dataset in datasets:
                # Try to download only if the downloader has a programmatic download method
                try:
                    # Heuristic: skip if the download method only logs a manual instruction
                    # We'll try, and if it raises NotImplementedError or logs a manual step, we skip
                    logger.info(f"Attempting to download {dataset.name} from {downloader.source_name}")
                    downloader.download(dataset.name, dest_dir)
                except NotImplementedError:
                    logger.info(f"Skipping {dataset.name} from {downloader.source_name}: not implemented for programmatic download.")
                except Exception as e:
                    logger.warning(f"Could not download {dataset.name} from {downloader.source_name}: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not list datasets for {downloader.source_name}: {str(e)}")
    logger.info("Auto-download complete.")

def main():
    """Main entry point for the data downloader.

    Examples:
        List all datasets:
            python download_data.py --list
        List genomics datasets:
            python download_data.py --list --type genomics
        Download a dataset:
            python download_data.py --source tcga --dataset tcga --dest data/raw/tcga
        Download a Kaggle dataset:
            python download_data.py --source kaggle --dataset awsaf49/cbis-ddsm-breast-cancer-image-dataset --dest data/raw/ddsm
    """
    parser = argparse.ArgumentParser(
        description="Download medical, genomics, imaging, and clinical datasets from major repositories. See README.md for full list of supported sources.",
        epilog="Examples:\n  python download_data.py --list\n  python download_data.py --list --type genomics\n  python download_data.py --source tcga --dataset tcga --dest data/raw/tcga\n  python download_data.py --source kaggle --dataset awsaf49/cbis-ddsm-breast-cancer-image-dataset --dest data/raw/ddsm"
    )
    parser.add_argument("--list", action="store_true", help="List all available datasets (optionally filter by type)")
    parser.add_argument("--type", choices=[t.value for t in DataType], help="Filter datasets by type: genomics, imaging, clinical, public")
    parser.add_argument("--source", help="Source name for downloading (see README for full list)")
    parser.add_argument("--dataset", help="Dataset name to download (see --list for options)")
    parser.add_argument("--dest", type=Path, help="Destination directory for downloads")
    parser.add_argument("--log", type=Path, help="Log file path")
    parser.add_argument("--auto", action="store_true", help="Automatically download all datasets that can be fetched programmatically.")
    args = parser.parse_args()
    if args.list:
        list_available_datasets(DataType(args.type) if args.type else None)
    elif args.source and args.dataset:
        download_dataset(args.source, args.dataset, args.dest)
    elif args.auto:
        auto_download_all(args.dest)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 