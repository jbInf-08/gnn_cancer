import os
import requests

os.makedirs('data/external/pathways', exist_ok=True)
url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human"
out_path = "data/external/pathways/KEGG_2021_Human.gmt"
r = requests.get(url)
with open(out_path, "wb") as f:
    f.write(r.content)
print("KEGG pathway data downloaded.") 