#!/usr/bin/env python3
"""
Delete corrupted files from the project
"""

import os
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_corrupted_files():
    """Delete all corrupted files found in the project"""
    
    # Files to delete
    files_to_delete = []
    
    # 1. Delete all .parcel files (corrupted log files)
    parcel_files = glob.glob("data/**/*.parcel", recursive=True)
    files_to_delete.extend(parcel_files)
    
    # 2. Delete empty files
    for root, dirs, files in os.walk("data"):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                files_to_delete.append(file_path)
    
    # 3. Delete very small files that are likely corrupted
    for root, dirs, files in os.walk("data"):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) < 100:  # Less than 100 bytes
                files_to_delete.append(file_path)
    
    # Remove duplicates
    files_to_delete = list(set(files_to_delete))
    
    # Delete files
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
    
    logger.info(f"Deleted {deleted_count} corrupted files")
    
    # Also clean up empty directories
    for root, dirs, files in os.walk("data", topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    logger.info(f"Deleted empty directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to delete directory {dir_path}: {e}")

if __name__ == "__main__":
    delete_corrupted_files() 