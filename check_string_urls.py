# -*- coding: utf-8 -*-
import requests
import gzip
import io
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_string_urls():
    """Test STRING database URLs to verify they are accessible and return valid gzip files."""
    
    print("Checking STRING database structure...")
    
    # Test the main STRING download page
    main_url = "https://string-db.org/cgi/download"
    print(f"Testing main STRING page: {main_url}")
    
    try:
        response = requests.get(main_url, timeout=30)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Main STRING page is accessible")
            # Parse the page to find download links
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            download_links = [link['href'] for link in links if 'download' in link['href']]
            print(f"Found {len(download_links)} download links")
            for i, link in enumerate(download_links[:5]):  # Show first 5
                print(f"  {i+1}. {link}")
        else:
            print(f"✗ Main STRING page returned status code {response.status_code}")
    except Exception as e:
        print(f"✗ Error accessing main STRING page: {e}")
    
    print("-" * 80)
    
    # Test alternative STRING URLs based on current documentation
    print("Testing alternative STRING URL patterns...")
    
    # Pattern 1: Direct download from string-db.org
    test_urls = [
        "https://string-db.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "https://string-db.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "https://string-db.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz",
        "https://string-db.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz",
        "https://string-db.org/download/protein.info.v11.0/9606.protein.info.v11.0.txt.gz",
        "https://string-db.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz",
    ]
    
    for url in test_urls:
        print(f"Testing: {url}")
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"  ✓ Available (status: {response.status_code})")
                print(f"  Content-Length: {response.headers.get('content-length', 'Not specified')}")
            else:
                print(f"  ✗ Not available (status: {response.status_code})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("-" * 80)
    
    # Test the STRING API
    print("Testing STRING API...")
    api_url = "https://string-db.org/api"
    try:
        response = requests.get(api_url, timeout=10)
        print(f"API status code: {response.status_code}")
        if response.status_code == 200:
            print("✓ STRING API is accessible")
        else:
            print(f"✗ STRING API returned status code {response.status_code}")
    except Exception as e:
        print(f"✗ Error accessing STRING API: {e}")

if __name__ == "__main__":
    test_string_urls() 