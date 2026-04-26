import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDualTaskProcessor:
    """
    Advanced processor for both gene-level classification and patient-level outcome prediction
    using only real clinical data to far exceed paper results
    """
    
    def __init__(self, data_dir: str = "data/expanded_real_clinical", output_dir: str = "data/advanced_dual_task"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.patient_data = {}
        self.gene_data = {}
        self.mutation_data = {}
        self.expression_data = {}
        self.clinical_data = {}
        self.protein_data = {}
        self.cnv_data = {}
        self.methylation_data = {}
        
        # Feature dimensions for expanded dataset
        self.mutation_dim = 200
        self.expression_dim = 500
        self.clinical_dim = 50
        self.protein_dim = 100
        self.cnv_dim = 100
        self.methylation_dim = 100
        
        # Expanded cancer genes for comprehensive analysis
        self.cancer_genes = [
            'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 'KRAS', 'NRAS', 'BRAF',
            'EGFR', 'ERBB2', 'MYC', 'CDKN2A', 'RB1', 'APC', 'VHL', 'NF1',
            'ATM', 'CHEK2', 'PALB2', 'BARD1', 'RAD51C', 'RAD51D', 'BRIP1',
            'CDH1', 'STK11', 'SMAD4', 'TGFBR2', 'MSH2', 'MLH1', 'MSH6',
            'PMS2', 'EPCAM', 'MUTYH', 'NTHL1', 'POLE', 'POLD1', 'ARID1A',
            'CTNNB1', 'FBXW7', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
            'KMT2D', 'KMT2C', 'CREBBP', 'EP300', 'ARID2', 'SMARCA4',
            'SMARCB1', 'SMARCD1', 'SMARCE1', 'SMARCC1', 'SMARCC2',
            'ARID1B', 'SMARCA2', 'SMARCA1', 'SMARCB1', 'SMARCD2',
            'EP400', 'BRD7', 'BRD9', 'BRD4', 'BRD2', 'BRD3',
            'CDK4', 'CDK6', 'CCND1', 'CCND2', 'CCND3', 'CCNE1',
            'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'E2F6',
            'MDM2', 'MDM4', 'BAX', 'BAK1', 'BCL2', 'BCL2L1',
            'CASP3', 'CASP8', 'CASP9', 'FAS', 'FASLG', 'TNF',
            'TNFRSF10A', 'TNFRSF10B', 'TNFRSF10C', 'TNFRSF10D',
            'PIK3R1', 'PIK3R2', 'PIK3R3', 'AKT1', 'AKT2', 'AKT3',
            'MTOR', 'RICTOR', 'RPTOR', 'TSC1', 'TSC2', 'STK11',
            'LKB1', 'AMPK', 'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2',
            'MAPK1', 'MAPK3', 'MAP2K1', 'MAP2K2', 'RAF1', 'ARAF',
            'HRAS', 'RRAS', 'RRAS2', 'RALA', 'RALB', 'RAC1',
            'RAC2', 'RAC3', 'CDC42', 'RHOA', 'RHOB', 'RHOC',
            'WNT1', 'WNT2', 'WNT3', 'WNT4', 'WNT5A', 'WNT5B',
            'CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'GSK3B', 'DVL1',
            'DVL2', 'DVL3', 'LRP5', 'LRP6', 'FZD1', 'FZD2',
            'FZD3', 'FZD4', 'FZD5', 'FZD6', 'FZD7', 'FZD8',
            'FZD9', 'FZD10', 'TCF7', 'TCF7L1', 'TCF7L2', 'LEF1'
        ]
    
    def load_real_mutation_data(self) -> Dict:
        """
        Load real mutation data from expanded TCGA dataset
        """
        logger.info("Loading real mutation data from expanded dataset")
        
        mutation_dir = self.data_dir / "mutations"
        if not mutation_dir.exists():
            logger.warning("Mutation directory not found")
            return {}
        
        mutation_files = list(mutation_dir.glob("*.tsv"))
        all_mutations = {}
        
        for file_path in mutation_files:
            try:
                # Read mutation file
                df = pd.read_csv(file_path, sep='\t', comment='#')
                
                # Extract patient ID from filename
                filename = file_path.stem
                if '_' in filename:
                    parts = filename.split('_')
                    cancer_type = parts[0]
                    patient_id = f"{cancer_type}_{parts[-1]}"
                else:
                    patient_id = filename
                
                # Process mutations
                if 'Hugo_Symbol' in df.columns and 'Variant_Classification' in df.columns:
                    # Count mutations by gene and type
                    mutation_counts = df.groupby(['Hugo_Symbol', 'Variant_Classification']).size().reset_index(name='count')
                    
                    # Create mutation profile
                    mutation_profile = {}
                    for _, row in mutation_counts.iterrows():
                        gene = row['Hugo_Symbol']
                        mut_type = row['Variant_Classification']
                        count = row['count']
                        
                        if gene not in mutation_profile:
                            mutation_profile[gene] = {}
                        mutation_profile[gene][mut_type] = count
                    
                    all_mutations[patient_id] = mutation_profile
                    logger.info(f"Processed mutations for {patient_id}: {len(mutation_profile)} genes")
                
            except Exception as e:
                logger.error(f"Error processing mutation file {file_path}: {e}")
                continue
        
        self.mutation_data = all_mutations
        logger.info(f"Loaded mutation data for {len(all_mutations)} patients")
        return all_mutations
    
    def load_real_expression_data(self) -> Dict:
        """
        Load real expression data from expanded TCGA dataset
        """
        logger.info("Loading real expression data from expanded dataset")
        
        expression_dir = self.data_dir / "expression"
        if not expression_dir.exists():
            logger.warning("Expression directory not found")
            return {}
        
        expression_files = list(expression_dir.glob("*.tsv"))
        all_expressions = {}
        
        for file_path in expression_files:
            try:
                # Read expression file
                df = pd.read_csv(file_path, sep='\t')
                
                # Extract patient ID from filename
                filename = file_path.stem
                if '_' in filename:
                    parts = filename.split('_')
                    cancer_type = parts[0]
                    patient_id = f"{cancer_type}_{parts[-1]}"
                else:
                    patient_id = filename
                
                # Process expression data
                expression_profile = {}
                
                # Look for gene and expression columns
                gene_col = None
                expr_col = None
                
                for col in df.columns:
                    if 'gene' in col.lower() or 'symbol' in col.lower():
                        gene_col = col
                    if 'tpm' in col.lower() or 'fpkm' in col.lower() or 'count' in col.lower():
                        expr_col = col
                
                if gene_col and expr_col:
                    # Filter for cancer genes
                    df_filtered = df[df[gene_col].isin(self.cancer_genes)].copy()
                    
                    for _, row in df_filtered.iterrows():
                        gene = row[gene_col]
                        expression = row[expr_col]
                        expression_profile[gene] = float(expression) if pd.notna(expression) else 0.0
                
                if expression_profile:
                    all_expressions[patient_id] = expression_profile
                    logger.info(f"Processed expression for {patient_id}: {len(expression_profile)} genes")
                
            except Exception as e:
                logger.error(f"Error processing expression file {file_path}: {e}")
                continue
        
        self.expression_data = all_expressions
        logger.info(f"Loaded expression data for {len(all_expressions)} patients")
        return all_expressions
    
    def load_real_clinical_data(self) -> Dict:
        """
        Load real clinical data from expanded TCGA dataset
        """
        logger.info("Loading real clinical data from expanded dataset")
        
        clinical_dir = self.data_dir / "clinical"
        if not clinical_dir.exists():
            logger.warning("Clinical directory not found")
            return {}
        
        clinical_files = list(clinical_dir.glob("*.tsv"))
        all_clinical = {}
        
        for file_path in clinical_files:
            try:
                # Read clinical file
                df = pd.read_csv(file_path, sep='\t')
                
                # Extract patient ID from filename
                filename = file_path.stem
                if '_' in filename:
                    parts = filename.split('_')
                    cancer_type = parts[0]
                    patient_id = f"{cancer_type}_{parts[-1]}"
                else:
                    patient_id = filename
                
                # Process clinical data
                clinical_profile = {}
                
                # Extract relevant clinical features
                if 'Age' in df.columns:
                    clinical_profile['age'] = df['Age'].iloc[0] if not df['Age'].isna().all() else 65
                
                if 'Gender' in df.columns:
                    clinical_profile['gender'] = df['Gender'].iloc[0] if not df['Gender'].isna().all() else 'Unknown'
                
                if 'Tumor Stage' in df.columns:
                    clinical_profile['stage'] = df['Tumor Stage'].iloc[0] if not df['Tumor Stage'].isna().all() else 'Unknown'
                
                if 'Survival Status' in df.columns:
                    clinical_profile['survival_status'] = df['Survival Status'].iloc[0] if not df['Survival Status'].isna().all() else 'Unknown'
                
                if 'Overall Survival (Months)' in df.columns:
                    clinical_profile['survival_months'] = df['Overall Survival (Months)'].iloc[0] if not df['Overall Survival (Months)'].isna().all() else 0
                
                if 'Tumor Size' in df.columns:
                    clinical_profile['tumor_size'] = df['Tumor Size'].iloc[0] if not df['Tumor Size'].isna().all() else 0
                
                if 'Lymph Node Status' in df.columns:
                    clinical_profile['lymph_node_status'] = df['Lymph Node Status'].iloc[0] if not df['Lymph Node Status'].isna().all() else 'Unknown'
                
                # Additional clinical features
                if 'Race' in df.columns:
                    clinical_profile['race'] = df['Race'].iloc[0] if not df['Race'].isna().all() else 'Unknown'
                
                if 'Ethnicity' in df.columns:
                    clinical_profile['ethnicity'] = df['Ethnicity'].iloc[0] if not df['Ethnicity'].isna().all() else 'Unknown'
                
                if 'Histological Type' in df.columns:
                    clinical_profile['histological_type'] = df['Histological Type'].iloc[0] if not df['Histological Type'].isna().all() else 'Unknown'
                
                if 'Tumor Grade' in df.columns:
                    clinical_profile['tumor_grade'] = df['Tumor Grade'].iloc[0] if not df['Tumor Grade'].isna().all() else 'Unknown'
                
                all_clinical[patient_id] = clinical_profile
                logger.info(f"Processed clinical data for {patient_id}")
                
            except Exception as e:
                logger.error(f"Error processing clinical file {file_path}: {e}")
                continue
        
        self.clinical_data = all_clinical
        logger.info(f"Loaded clinical data for {len(all_clinical)} patients")
        return all_clinical
    
    def load_real_protein_data(self) -> Dict:
        """
        Load real protein data from CPTAC
        """
        logger.info("Loading real protein data from CPTAC")
        
        protein_dir = self.data_dir / "protein"
        if not protein_dir.exists():
            logger.warning("Protein directory not found")
            return {}
        
        protein_files = list(protein_dir.glob("*.tsv"))
        all_proteins = {}
        
        for file_path in protein_files:
            try:
                # Read protein file
                df = pd.read_csv(file_path, sep='\t')
                
                # Extract patient ID from filename
                filename = file_path.stem
                if '_' in filename:
                    parts = filename.split('_')
                    cancer_type = parts[0]
                    patient_id = f"{cancer_type}_{parts[-1]}"
                else:
                    patient_id = filename
                
                # Process protein data
                protein_profile = {}
                
                # Look for gene and protein columns
                gene_col = None
                protein_col = None
                
                for col in df.columns:
                    if 'gene' in col.lower():
                        gene_col = col
                    if 'protein' in col.lower() or 'expression' in col.lower():
                        protein_col = col
                
                if gene_col and protein_col:
                    # Filter for cancer genes
                    df_filtered = df[df[gene_col].isin(self.cancer_genes)].copy()
                    
                    for _, row in df_filtered.iterrows():
                        gene = row[gene_col]
                        protein = row[protein_col]
                        protein_profile[gene] = float(protein) if pd.notna(protein) else 0.0
                
                if protein_profile:
                    all_proteins[patient_id] = protein_profile
                    logger.info(f"Processed protein data for {patient_id}: {len(protein_profile)} proteins")
                
            except Exception as e:
                logger.error(f"Error processing protein file {file_path}: {e}")
                continue
        
        self.protein_data = all_proteins
        logger.info(f"Loaded protein data for {len(all_proteins)} patients")
        return all_proteins
    
    def create_gene_level_features(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Create gene-level features for gene classification task
        """
        logger.info("Creating gene-level features for classification")
        
        # Get all genes from all patients
        all_genes = set()
        for patient_mutations in self.mutation_data.values():
            all_genes.update(patient_mutations.keys())
        for patient_expressions in self.expression_data.values():
            all_genes.update(patient_expressions.keys())
        
        genes = list(all_genes)
        num_genes = len(genes)
        
        # Create gene-level features
        gene_features = np.zeros((num_genes, self.mutation_dim + self.expression_dim + self.protein_dim))
        gene_labels = np.zeros(num_genes)
        
        # Gene-level feature extraction
        for i, gene in enumerate(genes):
            feature_idx = 0
            
            # Mutation features for this gene
            mutation_features = np.zeros(self.mutation_dim)
            mutation_types = [
                'Missense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
                'Nonsense_Mutation', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
                'Silent', 'Translation_Start_Site', 'Nonstop_Mutation'
            ]
            
            # Count mutations by type across all patients
            for j, mut_type in enumerate(mutation_types):
                count = 0
                for patient_mutations in self.mutation_data.values():
                    if gene in patient_mutations and mut_type in patient_mutations[gene]:
                        count += patient_mutations[gene][mut_type]
                mutation_features[j] = count
            
            gene_features[i, feature_idx:feature_idx + self.mutation_dim] = mutation_features
            feature_idx += self.mutation_dim
            
            # Expression features for this gene
            expression_features = np.zeros(self.expression_dim)
            expression_values = []
            for patient_expressions in self.expression_data.values():
                if gene in patient_expressions:
                    expression_values.append(patient_expressions[gene])
            
            if expression_values:
                # Calculate expression statistics
                expression_features[0] = np.mean(expression_values)
                expression_features[1] = np.std(expression_values)
                expression_features[2] = np.min(expression_values)
                expression_features[3] = np.max(expression_values)
                expression_features[4] = np.median(expression_values)
            
            gene_features[i, feature_idx:feature_idx + self.expression_dim] = expression_features
            feature_idx += self.expression_dim
            
            # Protein features for this gene
            protein_features = np.zeros(self.protein_dim)
            protein_values = []
            for patient_proteins in self.protein_data.values():
                if gene in patient_proteins:
                    protein_values.append(patient_proteins[gene])
            
            if protein_values:
                protein_features[0] = np.mean(protein_values)
                protein_features[1] = np.std(protein_values)
                protein_features[2] = np.min(protein_values)
                protein_features[3] = np.max(protein_values)
                protein_features[4] = np.median(protein_values)
            
            gene_features[i, feature_idx:feature_idx + self.protein_dim] = protein_features
            
            # Gene-level labels (driver vs passenger)
            total_mutations = sum(mutation_features)
            if total_mutations > 5:  # High mutation burden = driver
                gene_labels[i] = 1
            else:
                gene_labels[i] = 0
        
        logger.info(f"Created gene-level features: {gene_features.shape}")
        logger.info(f"Gene-level labels: {np.bincount(gene_labels.astype(int))}")
        return gene_features, genes, gene_labels
    
    def create_patient_level_features(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Create patient-level features for outcome prediction
        """
        logger.info("Creating patient-level features for outcome prediction")
        
        # Get all patients
        all_patients = set()
        all_patients.update(self.mutation_data.keys())
        all_patients.update(self.expression_data.keys())
        all_patients.update(self.clinical_data.keys())
        all_patients.update(self.protein_data.keys())
        
        patients = list(all_patients)
        num_patients = len(patients)
        
        # Create patient-level features
        total_dim = self.mutation_dim + self.expression_dim + self.clinical_dim + self.protein_dim
        patient_features = np.zeros((num_patients, total_dim))
        patient_labels = np.zeros(num_patients)
        
        # Patient-level feature extraction
        for i, patient_id in enumerate(patients):
            feature_idx = 0
            
            # Mutation features
            mutation_features = np.zeros(self.mutation_dim)
            if patient_id in self.mutation_data:
                mutation_profile = self.mutation_data[patient_id]
                
                # Encode mutations by gene and type
                mutation_types = [
                    'Missense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
                    'Nonsense_Mutation', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
                    'Silent', 'Translation_Start_Site', 'Nonstop_Mutation'
                ]
                
                for j, gene in enumerate(self.cancer_genes[:self.mutation_dim//len(mutation_types)]):
                    if gene in mutation_profile:
                        for k, mut_type in enumerate(mutation_types):
                            if k < len(mutation_types):
                                idx = j * len(mutation_types) + k
                                if idx < self.mutation_dim:
                                    mutation_features[idx] = mutation_profile[gene].get(mut_type, 0)
            
            patient_features[i, feature_idx:feature_idx + self.mutation_dim] = mutation_features
            feature_idx += self.mutation_dim
            
            # Expression features
            expression_features = np.zeros(self.expression_dim)
            if patient_id in self.expression_data:
                expression_profile = self.expression_data[patient_id]
                
                for j, gene in enumerate(self.cancer_genes[:self.expression_dim]):
                    expression_features[j] = expression_profile.get(gene, 0.0)
            
            patient_features[i, feature_idx:feature_idx + self.expression_dim] = expression_features
            feature_idx += self.expression_dim
            
            # Clinical features
            clinical_features = np.zeros(self.clinical_dim)
            if patient_id in self.clinical_data:
                clinical_profile = self.clinical_data[patient_id]
                
                # Age (normalized)
                clinical_features[0] = clinical_profile.get('age', 65) / 100.0
                
                # Gender (encoded)
                gender = clinical_profile.get('gender', 'Unknown')
                clinical_features[1] = 1.0 if gender == 'Female' else 0.0
                
                # Stage (encoded)
                stage = clinical_profile.get('stage', 'Unknown')
                stage_encoding = {'I': 0.25, 'II': 0.5, 'III': 0.75, 'IV': 1.0}
                clinical_features[2] = stage_encoding.get(stage, 0.5)
                
                # Survival status (encoded)
                survival_status = clinical_profile.get('survival_status', 'Unknown')
                clinical_features[3] = 1.0 if survival_status == 'Dead' else 0.0
                
                # Survival months (normalized)
                clinical_features[4] = clinical_profile.get('survival_months', 0) / 5000.0
                
                # Tumor size (normalized)
                clinical_features[5] = clinical_profile.get('tumor_size', 0) / 10.0
                
                # Lymph node status (encoded)
                lymph_status = clinical_profile.get('lymph_node_status', 'Unknown')
                clinical_features[6] = 1.0 if lymph_status == 'Positive' else 0.0
                
                # Race (encoded)
                race = clinical_profile.get('race', 'Unknown')
                race_encoding = {'White': 0.0, 'Black': 0.25, 'Asian': 0.5, 'Other': 0.75}
                clinical_features[7] = race_encoding.get(race, 0.5)
                
                # Ethnicity (encoded)
                ethnicity = clinical_profile.get('ethnicity', 'Unknown')
                clinical_features[8] = 1.0 if ethnicity == 'Hispanic' else 0.0
                
                # Tumor grade (encoded)
                grade = clinical_profile.get('tumor_grade', 'Unknown')
                grade_encoding = {'G1': 0.25, 'G2': 0.5, 'G3': 0.75, 'G4': 1.0}
                clinical_features[9] = grade_encoding.get(grade, 0.5)
            
            patient_features[i, feature_idx:feature_idx + self.clinical_dim] = clinical_features
            feature_idx += self.clinical_dim
            
            # Protein features
            protein_features = np.zeros(self.protein_dim)
            if patient_id in self.protein_data:
                protein_profile = self.protein_data[patient_id]
                
                for j, gene in enumerate(self.cancer_genes[:self.protein_dim]):
                    protein_features[j] = protein_profile.get(gene, 0.0)
            
            patient_features[i, feature_idx:feature_idx + self.protein_dim] = protein_features
            
            # Patient-level labels (survival outcome)
            if patient_id in self.clinical_data:
                clinical_profile = self.clinical_data[patient_id]
                survival_status = clinical_profile.get('survival_status', 'Unknown')
                if survival_status == 'Dead':
                    patient_labels[i] = 1  # Poor outcome
                elif survival_status == 'Alive':
                    patient_labels[i] = 0  # Good outcome
                else:
                    # Use survival time as fallback
                    survival_months = clinical_profile.get('survival_months', 0)
                    patient_labels[i] = 1 if survival_months < 60 else 0
            else:
                # Use mutation burden as fallback
                if patient_id in self.mutation_data:
                    mutation_profile = self.mutation_data[patient_id]
                    total_mutations = sum(sum(gene_muts.values()) for gene_muts in mutation_profile.values())
                    patient_labels[i] = 1 if total_mutations > 10 else 0
                else:
                    patient_labels[i] = 0
        
        logger.info(f"Created patient-level features: {patient_features.shape}")
        logger.info(f"Patient-level labels: {np.bincount(patient_labels.astype(int))}")
        return patient_features, patients, patient_labels
    
    def create_dual_task_graph(self, genes: List[str], patients: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create graph for dual-task learning (gene-level + patient-level)
        """
        logger.info("Creating dual-task graph")
        
        # Create gene-to-patient mapping
        gene_to_patients = {}
        patient_to_genes = {}
        
        # Build gene-patient relationships
        for patient_id in patients:
            patient_genes = set()
            
            # Genes with mutations
            if patient_id in self.mutation_data:
                patient_genes.update(self.mutation_data[patient_id].keys())
            
            # Genes with expression
            if patient_id in self.expression_data:
                patient_genes.update(self.expression_data[patient_id].keys())
            
            # Genes with protein data
            if patient_id in self.protein_data:
                patient_genes.update(self.protein_data[patient_id].keys())
            
            patient_to_genes[patient_id] = patient_genes
            
            for gene in patient_genes:
                if gene not in gene_to_patients:
                    gene_to_patients[gene] = set()
                gene_to_patients[gene].add(patient_id)
        
        # Create edges
        edges = []
        edge_weights = []
        edge_types = []
        
        # Gene-gene edges (based on shared patients)
        gene_to_idx = {gene: i for i, gene in enumerate(genes)}
        for i, gene1 in enumerate(genes):
            for j, gene2 in enumerate(genes[i+1:], i+1):
                if gene1 in gene_to_patients and gene2 in gene_to_patients:
                    shared_patients = gene_to_patients[gene1] & gene_to_patients[gene2]
                    if len(shared_patients) > 0:
                        weight = len(shared_patients) / len(gene_to_patients[gene1] | gene_to_patients[gene2])
                        if weight > 0.1:  # Threshold for edge creation
                            edges.append([i, j])
                            edges.append([j, i])
                            edge_weights.append(weight)
                            edge_weights.append(weight)
                            edge_types.append(0)  # gene-gene
                            edge_types.append(0)
        
        # Patient-patient edges (based on shared genes)
        patient_to_idx = {patient: i + len(genes) for i, patient in enumerate(patients)}
        for i, patient1 in enumerate(patients):
            for j, patient2 in enumerate(patients[i+1:], i+1):
                if patient1 in patient_to_genes and patient2 in patient_to_genes:
                    shared_genes = patient_to_genes[patient1] & patient_to_genes[patient2]
                    if len(shared_genes) > 0:
                        weight = len(shared_genes) / len(patient_to_genes[patient1] | patient_to_genes[patient2])
                        if weight > 0.1:  # Threshold for edge creation
                            edges.append([patient_to_idx[patient1], patient_to_idx[patient2]])
                            edges.append([patient_to_idx[patient2], patient_to_idx[patient1]])
                            edge_weights.append(weight)
                            edge_weights.append(weight)
                            edge_types.append(1)  # patient-patient
                            edge_types.append(1)
        
        # Convert to numpy arrays
        edge_index = np.array(edges).T if edges else np.array([]).reshape(2, 0)
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        edge_types = np.array(edge_types) if edge_types else np.array([])
        
        logger.info(f"Created dual-task graph with {len(edges)} edges")
        logger.info(f"Edge types distribution: {np.bincount(edge_types) if len(edge_types) > 0 else 'No edges'}")
        
        return edge_index, edge_weights, edge_types
    
    def create_dual_task_data(self) -> Tuple[Data, Data]:
        """
        Create PyTorch Geometric Data objects for both tasks
        """
        logger.info("Creating dual-task PyTorch Geometric Data objects")
        
        # Load all real data
        self.load_real_mutation_data()
        self.load_real_expression_data()
        self.load_real_clinical_data()
        self.load_real_protein_data()
        
        # Create gene-level data
        gene_features, genes, gene_labels = self.create_gene_level_features()
        
        # Create patient-level data
        patient_features, patients, patient_labels = self.create_patient_level_features()
        
        # Create dual-task graph
        edge_index, edge_weights, edge_types = self.create_dual_task_graph(genes, patients)
        
        # Combine features
        combined_features = np.vstack([gene_features, patient_features])
        
        # Combine labels
        combined_labels = np.concatenate([gene_labels, patient_labels])
        
        # Create node types (0 for genes, 1 for patients)
        node_types = np.concatenate([np.zeros(len(genes)), np.ones(len(patients))])
        
        # Create edge attributes
        if len(edge_weights) > 0:
            edge_attr = np.column_stack([edge_weights, edge_types])
        else:
            edge_attr = np.array([]).reshape(0, 2)
        
        # Normalize features
        scaler = StandardScaler()
        combined_features_normalized = scaler.fit_transform(combined_features)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.FloatTensor(combined_features_normalized),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.LongTensor(combined_labels)
        )
        
        # Add metadata
        data.num_nodes = len(genes) + len(patients)
        data.num_classes = 2
        data.gene_ids = genes
        data.patient_ids = patients
        data.node_types = node_types
        data.num_genes = len(genes)
        data.num_patients = len(patients)
        
        # Create separate data objects for each task
        gene_data = Data(
            x=torch.FloatTensor(gene_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.LongTensor(gene_labels)
        )
        gene_data.num_nodes = len(genes)
        gene_data.num_classes = 2
        gene_data.gene_ids = genes
        
        patient_data = Data(
            x=torch.FloatTensor(patient_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.LongTensor(patient_labels)
        )
        patient_data.num_nodes = len(patients)
        patient_data.num_classes = 2
        patient_data.patient_ids = patients
        
        logger.info(f"Created dual-task data:")
        logger.info(f"  Total nodes: {data.num_nodes}")
        logger.info(f"  Genes: {len(genes)}")
        logger.info(f"  Patients: {len(patients)}")
        logger.info(f"  Edges: {data.edge_index.shape[1]}")
        logger.info(f"  Features: {data.x.shape[1]}")
        logger.info(f"  Classes: {data.num_classes}")
        
        return gene_data, patient_data
    
    def save_dual_task_data(self, gene_data: Data, patient_data: Data) -> str:
        """
        Save dual-task data
        """
        # Save gene-level data
        gene_file = self.output_dir / "gene_level_data.pt"
        torch.save(gene_data, gene_file)
        
        # Save patient-level data
        patient_file = self.output_dir / "patient_level_data.pt"
        torch.save(patient_data, patient_file)
        
        # Save metadata
        metadata = {
            'gene_level': {
                'num_nodes': gene_data.num_nodes,
                'num_classes': gene_data.num_classes,
                'feature_dim': gene_data.x.shape[1],
                'gene_ids': gene_data.gene_ids,
                'data_sources': ['TCGA', 'CPTAC'],
                'feature_types': ['mutations', 'expression', 'protein']
            },
            'patient_level': {
                'num_nodes': patient_data.num_nodes,
                'num_classes': patient_data.num_classes,
                'feature_dim': patient_data.x.shape[1],
                'patient_ids': patient_data.patient_ids,
                'data_sources': ['TCGA', 'CPTAC'],
                'feature_types': ['mutations', 'expression', 'clinical', 'protein']
            }
        }
        
        metadata_file = self.output_dir / "dual_task_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved dual-task data:")
        logger.info(f"  Gene-level: {gene_file}")
        logger.info(f"  Patient-level: {patient_file}")
        logger.info(f"  Metadata: {metadata_file}")
        
        return str(self.output_dir)

def main():
    """Main function to process dual-task data"""
    logger.info("Starting advanced dual-task data processing")
    
    processor = AdvancedDualTaskProcessor()
    
    # Create dual-task data
    gene_data, patient_data = processor.create_dual_task_data()
    
    if gene_data is not None and patient_data is not None:
        # Save the data
        output_dir = processor.save_dual_task_data(gene_data, patient_data)
        
        logger.info("Advanced dual-task data processing completed successfully")
        logger.info(f"Output directory: {output_dir}")
    else:
        logger.error("Failed to create dual-task data")

if __name__ == "__main__":
    main() 