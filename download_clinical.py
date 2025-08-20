import os
import requests

os.makedirs('data/raw/tcga/BRCA/clinical', exist_ok=True)
url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv"
out_path = "data/raw/tcga/BRCA/clinical/TCGA-BRCA.survival.tsv"
r = requests.get(url)
with open(out_path, "wb") as f:
    f.write(r.content)
print("TCGA-BRCA clinical data downloaded.") 