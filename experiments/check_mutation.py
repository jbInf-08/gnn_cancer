import pandas as pd

# Check mutation file
df = pd.read_csv('data/raw/tcga/BRCA_mutation_6e6702e3-3950-4a23-a9c4-afefa989efed.maf', 
                 sep='\t', compression='gzip', comment='#')

print("Mutation file columns:")
print(df.columns.tolist())
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head()) 