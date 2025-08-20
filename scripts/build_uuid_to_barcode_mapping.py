import os
import re
import csv

def extract_uuid_barcode_pairs(tsv_path):
    uuid = None
    barcode = None
    pairs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            uuid_match = re.search(r'<shared:bcr_patient_uuid[^>]*>([a-f0-9\-]+)</shared:bcr_patient_uuid>', line, re.IGNORECASE)
            barcode_match = re.search(r'<shared:bcr_patient_barcode[^>]*>(TCGA-[A-Z0-9\-]+)</shared:bcr_patient_barcode>', line, re.IGNORECASE)
            if uuid_match:
                uuid = uuid_match.group(1).lower()
            if barcode_match:
                barcode = barcode_match.group(1)
            if uuid and barcode:
                pairs.append((uuid, barcode))
                uuid = None
                barcode = None
    return pairs

def main():
    clinical_dir = os.path.join('data', 'raw', 'tcga')
    mapping = []
    for fname in os.listdir(clinical_dir):
        if fname.startswith('BRCA_clinical_') and fname.endswith('.tsv'):
            path = os.path.join(clinical_dir, fname)
            pairs = extract_uuid_barcode_pairs(path)
            mapping.extend(pairs)
    # Remove duplicates
    mapping = list(set(mapping))
    # Write to CSV
    with open('uuid_to_barcode.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['uuid', 'barcode'])
        for uuid, barcode in mapping:
            writer.writerow([uuid, barcode])
    print(f"Wrote {len(mapping)} uuid-barcode pairs to uuid_to_barcode.csv")

if __name__ == '__main__':
    main() 