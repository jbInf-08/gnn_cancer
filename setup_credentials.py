import json
import os
from pathlib import Path
import webbrowser
import requests
from getpass import getpass

def setup_tcga():
    """Set up TCGA credentials and file IDs."""
    print("\n=== Setting up TCGA Access ===")
    print("1. Visit https://portal.gdc.cancer.gov/")
    print("2. Register for an account")
    print("3. After logging in, you can get your token from your profile")
    webbrowser.open("https://portal.gdc.cancer.gov/")
    
    token = getpass("Enter your TCGA token (optional, press Enter to skip): ")
    
    print("\nTo get file UUIDs:")
    print("1. Go to https://portal.gdc.cancer.gov/")
    print("2. Select your cancer type (e.g., BRCA)")
    print("3. Select data type (e.g., mutation, expression)")
    print("4. Add files to cart")
    print("5. Download the manifest file")
    print("6. The file IDs will be in the manifest")
    
    return token

def setup_kaggle():
    """Set up Kaggle API credentials."""
    print("\n=== Setting up Kaggle API ===")
    print("1. Visit https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download a kaggle.json file")
    webbrowser.open("https://www.kaggle.com/account")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    print(f"\nPlace your kaggle.json file in: {kaggle_dir}")
    input("Press Enter after you've placed the file...")

def main():
    config_file = Path("config/api_keys.json")
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {
            "TCGA_TOKEN": "",
            "TCGA_FILE_IDS": {
                "BRCA": {
                    "mutation": [],
                    "expression": [],
                    "cnv": []
                }
            }
        }
    
    # Set up TCGA
    tcga_token = setup_tcga()
    if tcga_token:
        config["TCGA_TOKEN"] = tcga_token
    
    # Set up Kaggle
    setup_kaggle()
    
    # Save configuration
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    
    print("\nConfiguration saved to config/api_keys.json")
    print("You can now run the downloader with these credentials.")

if __name__ == "__main__":
    main() 