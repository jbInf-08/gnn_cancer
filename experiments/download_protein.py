import os
import requests

os.makedirs('data/raw/tcga/BRCA/protein', exist_ok=True)
url = "https://github.com/chadcreighton/cancer-proteomics-compendium-n2002/raw/main/BREAST-CPTAC-TCGA_normalized_total_protein.xlsx"
out_path = "data/raw/tcga/BRCA/protein/BREAST-CPTAC-TCGA_normalized_total_protein.xlsx"
r = requests.get(url)
with open(out_path, "wb") as f:
    f.write(r.content)
print("Protein abundance data downloaded.") 