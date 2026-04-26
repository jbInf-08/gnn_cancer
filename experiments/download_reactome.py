import os
import requests

os.makedirs('data/external/pathways', exist_ok=True)
url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022"
out_path = "data/external/pathways/Reactome_2022.gmt"
r = requests.get(url)
with open(out_path, "wb") as f:
    f.write(r.content)
print("Reactome pathway data downloaded.") 