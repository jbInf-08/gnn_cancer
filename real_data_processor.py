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

class RealDataProcessor:
    """
    Process real clinical data from TCGA and CPTAC to create comprehensive datasets
    """
    
    def __init__(self, data_dir: str = "data/real_clinical"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/real_processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.patient_data = {}
        self.mutation_data = {}
        self.expression_data = {}
        self.clinical_data = {}
        self.protein_data = {}
        
        # Feature dimensions
        self.mutation_dim = 100
        self.expression_dim = 200
        self.clinical_dim = 20
        self.protein_dim = 50
        
        # Common cancer genes for feature alignment
        self.cancer_genes = [
            'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 'KRAS', 'NRAS', 'BRAF',
            'EGFR', 'ERBB2', 'MYC', 'CDKN2A', 'RB1', 'APC', 'VHL', 'NF1',
            'ATM', 'CHEK2', 'PALB2', 'BARD1', 'RAD51C', 'RAD51D', 'BRIP1',
            'CDH1', 'STK11', 'SMAD4', 'TGFBR2', 'MSH2', 'MLH1', 'MSH6',
            'PMS2', 'EPCAM', 'MUTYH', 'NTHL1', 'POLE', 'POLD1', 'ARID1A',
            'CTNNB1', 'FBXW7', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
            'KMT2D', 'KMT2C', 'CREBBP', 'EP300', 'ARID2', 'SMARCA4',
            'SMARCB1', 'SMARCD1', 'SMARCE1', 'SMARCC1', 'SMARCC2'
        ]
    
    def load_real_mutation_data(self) -> Dict:
        """
        Load and process real mutation data from TCGA
        """
        logger.info("Loading real mutation data")
        
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
        Load and process real expression data from TCGA
        """
        logger.info("Loading real expression data")
        
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
                if 'gene_id' in df.columns and 'tpm' in df.columns:
                    # Filter for cancer genes
                    df_filtered = df[df['gene_id'].isin(self.cancer_genes)].copy()
                    
                    # Create expression profile
                    expression_profile = {}
                    for _, row in df_filtered.iterrows():
                        gene = row['gene_id']
                        expression = row['tpm']
                        expression_profile[gene] = expression
                    
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
        Load and process real clinical data from TCGA
        """
        logger.info("Loading real clinical data")
        
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
        Load and process real protein data from CPTAC
        """
        logger.info("Loading real protein data")
        
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
                if 'Gene' in df.columns and 'Protein_Expression' in df.columns:
                    # Filter for cancer genes
                    df_filtered = df[df['Gene'].isin(self.cancer_genes)].copy()
                    
                    # Create protein profile
                    protein_profile = {}
                    for _, row in df_filtered.iterrows():
                        gene = row['Gene']
                        expression = row['Protein_Expression']
                        protein_profile[gene] = expression
                    
                    all_proteins[patient_id] = protein_profile
                    logger.info(f"Processed protein data for {patient_id}: {len(protein_profile)} proteins")
                
            except Exception as e:
                logger.error(f"Error processing protein file {file_path}: {e}")
                continue
        
        self.protein_data = all_proteins
        logger.info(f"Loaded protein data for {len(all_proteins)} patients")
        return all_proteins
    
    def create_real_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create comprehensive features from real clinical data
        """
        logger.info("Creating real clinical features")
        
        # Get all patients
        all_patients = set()
        all_patients.update(self.mutation_data.keys())
        all_patients.update(self.expression_data.keys())
        all_patients.update(self.clinical_data.keys())
        all_patients.update(self.protein_data.keys())
        
        if not all_patients:
            logger.error("No patient data available")
            return np.array([]), []
        
        patients = list(all_patients)
        total_dim = self.mutation_dim + self.expression_dim + self.clinical_dim + self.protein_dim
        
        features = np.zeros((len(patients), total_dim))
        
        # Mutation types for encoding
        mutation_types = [
            'Missense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
            'Nonsense_Mutation', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
            'Silent', 'Translation_Start_Site', 'Nonstop_Mutation'
        ]
        
        for i, patient_id in enumerate(patients):
            feature_idx = 0
            
            # Mutation features
            mutation_features = np.zeros(self.mutation_dim)
            if patient_id in self.mutation_data:
                mutation_profile = self.mutation_data[patient_id]
                
                # Encode mutations by gene and type
                for j, gene in enumerate(self.cancer_genes[:self.mutation_dim//len(mutation_types)]):
                    if gene in mutation_profile:
                        for k, mut_type in enumerate(mutation_types):
                            if k < len(mutation_types):
                                idx = j * len(mutation_types) + k
                                if idx < self.mutation_dim:
                                    mutation_features[idx] = mutation_profile[gene].get(mut_type, 0)
            
            features[i, feature_idx:feature_idx + self.mutation_dim] = mutation_features
            feature_idx += self.mutation_dim
            
            # Expression features
            expression_features = np.zeros(self.expression_dim)
            if patient_id in self.expression_data:
                expression_profile = self.expression_data[patient_id]
                
                for j, gene in enumerate(self.cancer_genes[:self.expression_dim]):
                    expression_features[j] = expression_profile.get(gene, 0.0)
            
            features[i, feature_idx:feature_idx + self.expression_dim] = expression_features
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
            
            features[i, feature_idx:feature_idx + self.clinical_dim] = clinical_features
            feature_idx += self.clinical_dim
            
            # Protein features
            protein_features = np.zeros(self.protein_dim)
            if patient_id in self.protein_data:
                protein_profile = self.protein_data[patient_id]
                
                for j, gene in enumerate(self.cancer_genes[:self.protein_dim]):
                    protein_features[j] = protein_profile.get(gene, 0.0)
            
            features[i, feature_idx:feature_idx + self.protein_dim] = protein_features
        
        logger.info(f"Created real features: {features.shape}")
        return features, patients
    
    def create_real_labels(self, patients: List[str]) -> np.ndarray:
        """
        Create real clinical outcome labels
        """
        logger.info("Creating real clinical outcome labels")
        
        labels = []
        for patient_id in patients:
            if patient_id in self.clinical_data:
                clinical_profile = self.clinical_data[patient_id]
                
                # Use survival status as primary label
                survival_status = clinical_profile.get('survival_status', 'Unknown')
                if survival_status == 'Dead':
                    label = 1  # High risk
                elif survival_status == 'Alive':
                    label = 0  # Low risk
                else:
                    # Use survival time as fallback
                    survival_months = clinical_profile.get('survival_months', 0)
                    label = 1 if survival_months < 1000 else 0  # High risk if < 1000 months
            else:
                # Use mutation burden as fallback
                if patient_id in self.mutation_data:
                    mutation_profile = self.mutation_data[patient_id]
                    total_mutations = sum(sum(gene_muts.values()) for gene_muts in mutation_profile.values())
                    label = 1 if total_mutations > 10 else 0  # High risk if > 10 mutations
                else:
                    label = 0  # Default to low risk
            
            labels.append(label)
        
        labels = np.array(labels)
        logger.info(f"Created real labels: {np.bincount(labels) if len(labels) > 0 else 'No labels'}")
        return labels
    
    def create_real_graph(self, patients: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create graph from real clinical data
        """
        logger.info("Creating real clinical graph")
        
        n_patients = len(patients)
        edges = []
        edge_weights = []
        edge_types = []
        
        # Create patient-to-index mapping
        patient_to_idx = {patient: i for i, patient in enumerate(patients)}
        
        # 1. Mutation similarity edges
        logger.info("Creating mutation similarity edges")
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                patient_i = patients[i]
                patient_j = patients[j]
                
                # Calculate mutation similarity
                similarity = self._calculate_mutation_similarity(patient_i, patient_j)
                if similarity > 0.2:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
                    edge_weights.append(similarity)
                    edge_weights.append(similarity)
                    edge_types.append(0)  # mutation_similarity
                    edge_types.append(0)
        
        # 2. Expression correlation edges
        logger.info("Creating expression correlation edges")
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                patient_i = patients[i]
                patient_j = patients[j]
                
                correlation = self._calculate_expression_correlation(patient_i, patient_j)
                if abs(correlation) > 0.4:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.append(abs(correlation))
                    edge_weights.append(abs(correlation))
                    edge_types.append(1)  # expression_correlation
                    edge_types.append(1)
        
        # 3. Clinical similarity edges
        logger.info("Creating clinical similarity edges")
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                patient_i = patients[i]
                patient_j = patients[j]
                
                similarity = self._calculate_clinical_similarity(patient_i, patient_j)
                if similarity > 0.3:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.append(similarity)
                    edge_weights.append(similarity)
                    edge_types.append(2)  # clinical_similarity
                    edge_types.append(2)
        
        # 4. Protein correlation edges
        logger.info("Creating protein correlation edges")
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                patient_i = patients[i]
                patient_j = patients[j]
                
                correlation = self._calculate_protein_correlation(patient_i, patient_j)
                if abs(correlation) > 0.3:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.append(abs(correlation))
                    edge_weights.append(abs(correlation))
                    edge_types.append(3)  # protein_correlation
                    edge_types.append(3)
        
        # Convert to numpy arrays
        edge_index = np.array(edges).T if edges else np.array([]).reshape(2, 0)
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        edge_types = np.array(edge_types) if edge_types else np.array([])
        
        logger.info(f"Created real graph with {len(edges)} edges")
        logger.info(f"Edge types distribution: {np.bincount(edge_types) if len(edge_types) > 0 else 'No edges'}")
        
        return edge_index, edge_weights, edge_types
    
    def _calculate_mutation_similarity(self, patient1: str, patient2: str) -> float:
        """Calculate mutation similarity between two patients"""
        if patient1 not in self.mutation_data or patient2 not in self.mutation_data:
            return 0.0
        
        mut1 = self.mutation_data[patient1]
        mut2 = self.mutation_data[patient2]
        
        if not mut1 or not mut2:
            return 0.0
        
        # Jaccard similarity
        all_genes = set(mut1.keys()) | set(mut2.keys())
        if not all_genes:
            return 0.0
        
        intersection = len(set(mut1.keys()) & set(mut2.keys()))
        union = len(all_genes)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_expression_correlation(self, patient1: str, patient2: str) -> float:
        """Calculate expression correlation between two patients"""
        if patient1 not in self.expression_data or patient2 not in self.expression_data:
            return 0.0
        
        expr1 = self.expression_data[patient1]
        expr2 = self.expression_data[patient2]
        
        if not expr1 or not expr2:
            return 0.0
        
        # Find common genes
        common_genes = set(expr1.keys()) & set(expr2.keys())
        if len(common_genes) < 5:
            return 0.0
        
        # Calculate correlation
        values1 = [expr1[gene] for gene in common_genes]
        values2 = [expr2[gene] for gene in common_genes]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def _calculate_clinical_similarity(self, patient1: str, patient2: str) -> float:
        """Calculate clinical similarity between two patients"""
        if patient1 not in self.clinical_data or patient2 not in self.clinical_data:
            return 0.0
        
        clin1 = self.clinical_data[patient1]
        clin2 = self.clinical_data[patient2]
        
        if not clin1 or not clin2:
            return 0.0
        
        # Calculate similarity based on clinical features
        similarities = []
        
        # Age similarity (normalized)
        if 'age' in clin1 and 'age' in clin2:
            age_diff = abs(clin1['age'] - clin2['age']) / 100.0
            similarities.append(1.0 - age_diff)
        
        # Gender similarity
        if 'gender' in clin1 and 'gender' in clin2:
            similarities.append(1.0 if clin1['gender'] == clin2['gender'] else 0.0)
        
        # Stage similarity
        if 'stage' in clin1 and 'stage' in clin2:
            similarities.append(1.0 if clin1['stage'] == clin2['stage'] else 0.0)
        
        # Survival status similarity
        if 'survival_status' in clin1 and 'survival_status' in clin2:
            similarities.append(1.0 if clin1['survival_status'] == clin2['survival_status'] else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_protein_correlation(self, patient1: str, patient2: str) -> float:
        """Calculate protein correlation between two patients"""
        if patient1 not in self.protein_data or patient2 not in self.protein_data:
            return 0.0
        
        prot1 = self.protein_data[patient1]
        prot2 = self.protein_data[patient2]
        
        if not prot1 or not prot2:
            return 0.0
        
        # Find common proteins
        common_proteins = set(prot1.keys()) & set(prot2.keys())
        if len(common_proteins) < 3:
            return 0.0
        
        # Calculate correlation
        values1 = [prot1[protein] for protein in common_proteins]
        values2 = [prot2[protein] for protein in common_proteins]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def create_pytorch_geometric_data(self) -> Data:
        """
        Create PyTorch Geometric Data object with real clinical data
        """
        logger.info("Creating PyTorch Geometric Data object with real clinical data")
        
        # Load all real data
        self.load_real_mutation_data()
        self.load_real_expression_data()
        self.load_real_clinical_data()
        self.load_real_protein_data()
        
        # Create features and get patient list
        features, patients = self.create_real_features()
        
        if len(patients) == 0:
            logger.error("No patients found in real data")
            return None
        
        # Create real graph
        edge_index, edge_weights, edge_types = self.create_real_graph(patients)
        
        # Create real labels
        labels = self.create_real_labels(patients)
        
        # Create edge attributes (combine weights and types)
        if len(edge_weights) > 0:
            edge_attr = np.column_stack([edge_weights, edge_types])
        else:
            edge_attr = np.array([]).reshape(0, 2)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.FloatTensor(features_normalized),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.LongTensor(labels)
        )
        
        # Add metadata
        data.num_nodes = len(patients)
        data.num_classes = len(np.unique(labels))
        data.patient_ids = patients
        
        logger.info(f"Created PyTorch Geometric Data with real clinical data:")
        logger.info(f"  Nodes: {data.num_nodes}")
        logger.info(f"  Edges: {data.edge_index.shape[1]}")
        logger.info(f"  Features: {data.x.shape[1]}")
        logger.info(f"  Classes: {data.num_classes}")
        logger.info(f"  Edge attributes: {data.edge_attr.shape[1] if len(data.edge_attr) > 0 else 0}")
        
        return data
    
    def save_real_data(self, data: Data) -> str:
        """
        Save real clinical data
        """
        if data is None:
            logger.error("No data to save")
            return ""
        
        output_file = self.output_dir / "real_clinical_data.pt"
        torch.save(data, output_file)
        
        # Save metadata
        metadata = {
            'num_nodes': data.num_nodes,
            'num_classes': data.num_classes,
            'feature_dim': data.x.shape[1],
            'patient_ids': data.patient_ids,
            'data_sources': ['TCGA', 'CPTAC'],
            'feature_types': ['mutations', 'expression', 'clinical', 'protein']
        }
        
        metadata_file = self.output_dir / "real_clinical_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved real clinical data to {output_file}")
        logger.info(f"Saved metadata to {metadata_file}")
        
        return str(output_file)

def main():
    """Main function to process real clinical data"""
    logger.info("Starting real clinical data processing")
    
    processor = RealDataProcessor()
    
    # Create PyTorch Geometric Data object with real data
    data = processor.create_pytorch_geometric_data()
    
    if data is not None:
        # Save the data
        output_file = processor.save_real_data(data)
        
        logger.info("Real clinical data processing completed successfully")
        logger.info(f"Output file: {output_file}")
    else:
        logger.error("Failed to create real clinical data")

if __name__ == "__main__":
    main() 