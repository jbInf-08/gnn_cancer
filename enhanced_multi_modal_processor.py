import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMultiModalProcessor:
    """
    Enhanced multi-modal processor that incorporates:
    - Protein abundance data
    - Metabolite levels
    - Clinical variables
    - Advanced graph construction with multiple edge types
    - Sophisticated edge weights
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/enhanced_multi_modal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.patient_data = {}
        self.protein_data = {}
        self.metabolite_data = {}
        self.clinical_data = {}
        self.ppi_network = None
        self.pathway_data = None
        
        # Feature dimensions
        self.mutation_dim = 50
        self.expression_dim = 100
        self.cnv_dim = 30
        self.protein_dim = 40
        self.metabolite_dim = 20
        self.clinical_dim = 15
        
        # Edge types and weights
        self.edge_types = {
            'mutation_similarity': 1.0,
            'expression_correlation': 0.8,
            'clinical_similarity': 0.6,
            'protein_correlation': 0.7,
            'metabolite_correlation': 0.5,
            'survival_similarity': 0.9,
            'tumor_stage_similarity': 0.8,
            'age_similarity': 0.4
        }
    
    def load_all_data(self) -> Dict:
        """
        Load all available data types
        """
        logger.info("Loading all multi-modal data")
        
        results = {
            'mutations': self._load_mutation_data(),
            'expression': self._load_expression_data(),
            'cnv': self._load_cnv_data(),
            'protein': self._load_protein_data(),
            'metabolite': self._load_metabolite_data(),
            'clinical': self._load_clinical_data(),
            'ppi': self._load_ppi_data(),
            'pathway': self._load_pathway_data()
        }
        
        logger.info(f"Loaded data summary:")
        for data_type, count in results.items():
            if isinstance(count, int):
                logger.info(f"  {data_type}: {count} files")
            else:
                logger.info(f"  {data_type}: {len(count) if count else 0} files")
        
        return results
    
    def _load_mutation_data(self) -> int:
        """Load mutation data from all available files"""
        mutation_dir = self.data_dir / "mutations"
        if not mutation_dir.exists():
            logger.warning("Mutation directory not found")
            return 0
        
        files = list(mutation_dir.glob("*.tsv")) + list(mutation_dir.glob("*.maf"))
        count = 0
        
        for file_path in files:
            try:
                if file_path.suffix == '.maf':
                    df = pd.read_csv(file_path, sep='\t', comment='#')
                else:
                    df = pd.read_csv(file_path, sep='\t')
                
                # Extract patient ID from filename or Tumor_Sample_Barcode
                if 'Tumor_Sample_Barcode' in df.columns:
                    patient_id = df['Tumor_Sample_Barcode'].iloc[0]
                else:
                    patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Count mutations by type
                mutation_counts = df['Variant_Classification'].value_counts().to_dict()
                self.patient_data[patient_id]['mutations'] = mutation_counts
                count += 1
                
            except Exception as e:
                logger.warning(f"Error loading mutation file {file_path}: {e}")
                continue
        
        return count
    
    def _load_expression_data(self) -> int:
        """Load expression data"""
        expression_dir = self.data_dir / "expression"
        if not expression_dir.exists():
            logger.warning("Expression directory not found")
            return 0
        
        files = list(expression_dir.glob("*.tsv"))
        count = 0
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Store expression data
                if 'Gene' in df.columns and 'Expression' in df.columns:
                    expression_dict = dict(zip(df['Gene'], df['Expression']))
                    self.patient_data[patient_id]['expression'] = expression_dict
                    count += 1
                
            except Exception as e:
                logger.warning(f"Error loading expression file {file_path}: {e}")
                continue
        
        return count
    
    def _load_cnv_data(self) -> int:
        """Load CNV data"""
        cnv_dir = self.data_dir / "cnv"
        if not cnv_dir.exists():
            logger.warning("CNV directory not found")
            return 0
        
        files = list(cnv_dir.glob("*.tsv"))
        count = 0
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Store CNV data
                if 'Gene' in df.columns and 'CNV_Value' in df.columns:
                    cnv_dict = dict(zip(df['Gene'], df['CNV_Value']))
                    self.patient_data[patient_id]['cnv'] = cnv_dict
                    count += 1
                
            except Exception as e:
                logger.warning(f"Error loading CNV file {file_path}: {e}")
                continue
        
        return count
    
    def _load_protein_data(self) -> int:
        """Load protein abundance data"""
        protein_dir = self.data_dir / "protein"
        if not protein_dir.exists():
            logger.warning("Protein directory not found")
            return 0
        
        files = list(protein_dir.glob("*.tsv"))
        count = 0
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Store protein data
                if 'Protein' in df.columns and 'Abundance' in df.columns:
                    protein_dict = dict(zip(df['Protein'], df['Abundance']))
                    self.patient_data[patient_id]['protein'] = protein_dict
                    self.protein_data[patient_id] = protein_dict
                    count += 1
                
            except Exception as e:
                logger.warning(f"Error loading protein file {file_path}: {e}")
                continue
        
        return count
    
    def _load_metabolite_data(self) -> int:
        """Load metabolite data"""
        metabolite_dir = self.data_dir / "metabolite"
        if not metabolite_dir.exists():
            logger.warning("Metabolite directory not found")
            return 0
        
        files = list(metabolite_dir.glob("*.tsv"))
        count = 0
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Store metabolite data
                if 'Metabolite' in df.columns and 'Concentration' in df.columns:
                    metabolite_dict = dict(zip(df['Metabolite'], df['Concentration']))
                    self.patient_data[patient_id]['metabolite'] = metabolite_dict
                    self.metabolite_data[patient_id] = metabolite_dict
                    count += 1
                
            except Exception as e:
                logger.warning(f"Error loading metabolite file {file_path}: {e}")
                continue
        
        return count
    
    def _load_clinical_data(self) -> int:
        """Load clinical data"""
        clinical_dir = self.data_dir / "clinical"
        if not clinical_dir.exists():
            logger.warning("Clinical directory not found")
            return 0
        
        files = list(clinical_dir.glob("*.tsv"))
        count = 0
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                patient_id = file_path.stem.split('_')[0]
                
                if patient_id not in self.patient_data:
                    self.patient_data[patient_id] = {}
                
                # Store clinical data
                clinical_dict = {}
                for col in df.columns:
                    if col != 'Patient_ID':
                        clinical_dict[col] = df[col].iloc[0]
                
                self.patient_data[patient_id]['clinical'] = clinical_dict
                self.clinical_data[patient_id] = clinical_dict
                count += 1
                
            except Exception as e:
                logger.warning(f"Error loading clinical file {file_path}: {e}")
                continue
        
        return count
    
    def _load_ppi_data(self) -> bool:
        """Load PPI network data"""
        ppi_file = self.data_dir / "string_ppi_network.tsv"
        if not ppi_file.exists():
            logger.warning("PPI network file not found")
            return False
        
        try:
            df = pd.read_csv(ppi_file, sep='\t')
            self.ppi_network = df
            logger.info(f"Loaded PPI network with {len(df)} interactions")
            return True
        except Exception as e:
            logger.error(f"Error loading PPI data: {e}")
            return False
    
    def _load_pathway_data(self) -> bool:
        """Load pathway data"""
        pathway_file = self.data_dir / "kegg_cancer_pathways.tsv"
        if not pathway_file.exists():
            logger.warning("Pathway file not found")
            return False
        
        try:
            df = pd.read_csv(pathway_file, sep='\t')
            self.pathway_data = df
            logger.info(f"Loaded pathway data with {len(df)} pathways")
            return True
        except Exception as e:
            logger.error(f"Error loading pathway data: {e}")
            return False
    
    def create_multi_modal_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create comprehensive multi-modal features for each patient
        """
        logger.info("Creating multi-modal features")
        
        patients = list(self.patient_data.keys())
        if not patients:
            logger.error("No patient data available")
            return np.array([]), []
        
        # Initialize feature matrix
        total_dim = (self.mutation_dim + self.expression_dim + self.cnv_dim + 
                    self.protein_dim + self.metabolite_dim + self.clinical_dim)
        
        features = np.zeros((len(patients), total_dim))
        
        # Common genes/proteins for feature alignment
        common_genes = ['TP53', 'BRCA1', 'BRCA2', 'PIK3CA', 'KRAS', 'BRAF', 'EGFR', 'MYC']
        common_proteins = ['TP53', 'BRCA1', 'BRCA2', 'PIK3CA', 'KRAS', 'BRAF', 'EGFR', 'MYC']
        common_metabolites = ['Glucose', 'Lactate', 'Glutamine', 'Glutamate', 'Citrate', 'Succinate']
        
        for i, patient_id in enumerate(patients):
            patient_data = self.patient_data[patient_id]
            feature_idx = 0
            
            # Mutation features (one-hot encoding of mutation types)
            mutation_features = np.zeros(self.mutation_dim)
            if 'mutations' in patient_data:
                mutation_counts = patient_data['mutations']
                mutation_types = ['Missense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 
                                'Nonsense_Mutation', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins']
                for j, mut_type in enumerate(mutation_types):
                    if j < self.mutation_dim:
                        mutation_features[j] = mutation_counts.get(mut_type, 0)
            features[i, feature_idx:feature_idx + self.mutation_dim] = mutation_features
            feature_idx += self.mutation_dim
            
            # Expression features
            expression_features = np.zeros(self.expression_dim)
            if 'expression' in patient_data:
                expr_data = patient_data['expression']
                for j, gene in enumerate(common_genes):
                    if j < self.expression_dim:
                        expression_features[j] = expr_data.get(gene, 0.0)
            features[i, feature_idx:feature_idx + self.expression_dim] = expression_features
            feature_idx += self.expression_dim
            
            # CNV features
            cnv_features = np.zeros(self.cnv_dim)
            if 'cnv' in patient_data:
                cnv_data = patient_data['cnv']
                for j, gene in enumerate(common_genes):
                    if j < self.cnv_dim:
                        cnv_features[j] = cnv_data.get(gene, 0.0)
            features[i, feature_idx:feature_idx + self.cnv_dim] = cnv_features
            feature_idx += self.cnv_dim
            
            # Protein features
            protein_features = np.zeros(self.protein_dim)
            if 'protein' in patient_data:
                protein_data = patient_data['protein']
                for j, protein in enumerate(common_proteins):
                    if j < self.protein_dim:
                        protein_features[j] = protein_data.get(protein, 0.0)
            features[i, feature_idx:feature_idx + self.protein_dim] = protein_features
            feature_idx += self.protein_dim
            
            # Metabolite features
            metabolite_features = np.zeros(self.metabolite_dim)
            if 'metabolite' in patient_data:
                metabolite_data = patient_data['metabolite']
                for j, metabolite in enumerate(common_metabolites):
                    if j < self.metabolite_dim:
                        metabolite_features[j] = metabolite_data.get(metabolite, 0.0)
            features[i, feature_idx:feature_idx + self.metabolite_dim] = metabolite_features
            feature_idx += self.metabolite_dim
            
            # Clinical features
            clinical_features = np.zeros(self.clinical_dim)
            if 'clinical' in patient_data:
                clinical_data = patient_data['clinical']
                clinical_features[0] = clinical_data.get('Age', 0) / 100.0  # Normalize age
                clinical_features[1] = 1.0 if clinical_data.get('Gender') == 'Female' else 0.0
                clinical_features[2] = clinical_data.get('Tumor_Size', 0) / 10.0  # Normalize tumor size
                clinical_features[3] = clinical_data.get('Survival_Time', 0) / 5000.0  # Normalize survival time
                clinical_features[4] = 1.0 if clinical_data.get('Lymph_Node_Status') == 'Positive' else 0.0
            features[i, feature_idx:feature_idx + self.clinical_dim] = clinical_features
        
        logger.info(f"Created multi-modal features: {features.shape}")
        return features, patients
    
    def create_advanced_graph(self, patients: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create advanced graph with multiple edge types and sophisticated weights
        """
        logger.info("Creating advanced graph with multiple edge types")
        
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
                if similarity > 0.3:  # Threshold for edge creation
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
                if abs(correlation) > 0.5:  # Threshold for edge creation
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
                if similarity > 0.4:  # Threshold for edge creation
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
                if abs(correlation) > 0.4:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.append(abs(correlation))
                    edge_weights.append(abs(correlation))
                    edge_types.append(3)  # protein_correlation
                    edge_types.append(3)
        
        # 5. Metabolite correlation edges
        logger.info("Creating metabolite correlation edges")
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                patient_i = patients[i]
                patient_j = patients[j]
                
                correlation = self._calculate_metabolite_correlation(patient_i, patient_j)
                if abs(correlation) > 0.3:  # Threshold for edge creation
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.append(abs(correlation))
                    edge_weights.append(abs(correlation))
                    edge_types.append(4)  # metabolite_correlation
                    edge_types.append(4)
        
        # Convert to numpy arrays
        edge_index = np.array(edges).T if edges else np.array([]).reshape(2, 0)
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])
        edge_types = np.array(edge_types) if edge_types else np.array([])
        
        logger.info(f"Created advanced graph with {len(edges)} edges")
        logger.info(f"Edge types distribution: {np.bincount(edge_types) if len(edge_types) > 0 else 'No edges'}")
        
        return edge_index, edge_weights, edge_types
    
    def _calculate_mutation_similarity(self, patient1: str, patient2: str) -> float:
        """Calculate mutation similarity between two patients"""
        if patient1 not in self.patient_data or patient2 not in self.patient_data:
            return 0.0
        
        mut1 = self.patient_data[patient1].get('mutations', {})
        mut2 = self.patient_data[patient2].get('mutations', {})
        
        if not mut1 or not mut2:
            return 0.0
        
        # Jaccard similarity
        all_mutations = set(mut1.keys()) | set(mut2.keys())
        if not all_mutations:
            return 0.0
        
        intersection = sum(min(mut1.get(mut, 0), mut2.get(mut, 0)) for mut in all_mutations)
        union = sum(max(mut1.get(mut, 0), mut2.get(mut, 0)) for mut in all_mutations)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_expression_correlation(self, patient1: str, patient2: str) -> float:
        """Calculate expression correlation between two patients"""
        if patient1 not in self.patient_data or patient2 not in self.patient_data:
            return 0.0
        
        expr1 = self.patient_data[patient1].get('expression', {})
        expr2 = self.patient_data[patient2].get('expression', {})
        
        if not expr1 or not expr2:
            return 0.0
        
        # Find common genes
        common_genes = set(expr1.keys()) & set(expr2.keys())
        if len(common_genes) < 3:
            return 0.0
        
        # Calculate correlation
        values1 = [expr1[gene] for gene in common_genes]
        values2 = [expr2[gene] for gene in common_genes]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def _calculate_clinical_similarity(self, patient1: str, patient2: str) -> float:
        """Calculate clinical similarity between two patients"""
        if patient1 not in self.patient_data or patient2 not in self.patient_data:
            return 0.0
        
        clin1 = self.patient_data[patient1].get('clinical', {})
        clin2 = self.patient_data[patient2].get('clinical', {})
        
        if not clin1 or not clin2:
            return 0.0
        
        # Calculate similarity based on clinical features
        similarities = []
        
        # Age similarity (normalized)
        if 'Age' in clin1 and 'Age' in clin2:
            age_diff = abs(clin1['Age'] - clin2['Age']) / 100.0
            similarities.append(1.0 - age_diff)
        
        # Gender similarity
        if 'Gender' in clin1 and 'Gender' in clin2:
            similarities.append(1.0 if clin1['Gender'] == clin2['Gender'] else 0.0)
        
        # Stage similarity
        if 'Stage' in clin1 and 'Stage' in clin2:
            similarities.append(1.0 if clin1['Stage'] == clin2['Stage'] else 0.0)
        
        # Tumor size similarity (normalized)
        if 'Tumor_Size' in clin1 and 'Tumor_Size' in clin2:
            size_diff = abs(clin1['Tumor_Size'] - clin2['Tumor_Size']) / 10.0
            similarities.append(1.0 - size_diff)
        
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
    
    def _calculate_metabolite_correlation(self, patient1: str, patient2: str) -> float:
        """Calculate metabolite correlation between two patients"""
        if patient1 not in self.metabolite_data or patient2 not in self.metabolite_data:
            return 0.0
        
        met1 = self.metabolite_data[patient1]
        met2 = self.metabolite_data[patient2]
        
        if not met1 or not met2:
            return 0.0
        
        # Find common metabolites
        common_metabolites = set(met1.keys()) & set(met2.keys())
        if len(common_metabolites) < 2:
            return 0.0
        
        # Calculate correlation
        values1 = [met1[metabolite] for metabolite in common_metabolites]
        values2 = [met2[metabolite] for metabolite in common_metabolites]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def create_labels(self, patients: List[str]) -> np.ndarray:
        """
        Create labels based on clinical outcomes or mutation burden
        """
        logger.info("Creating labels for patients")
        
        labels = []
        for patient_id in patients:
            if patient_id in self.patient_data and 'clinical' in self.patient_data[patient_id]:
                clinical_data = self.patient_data[patient_id]['clinical']
                
                # Try to use survival status as label
                if 'Survival_Status' in clinical_data:
                    label = 1 if clinical_data['Survival_Status'] == 'Dead' else 0
                elif 'Survival_Time' in clinical_data:
                    # Use survival time as proxy (high risk if < 1000 days)
                    label = 1 if clinical_data['Survival_Time'] < 1000 else 0
                else:
                    # Use mutation burden as fallback
                    mutation_count = sum(self.patient_data[patient_id].get('mutations', {}).values())
                    label = 1 if mutation_count > 10 else 0
            else:
                # Use mutation burden as fallback
                mutation_count = sum(self.patient_data[patient_id].get('mutations', {}).values())
                label = 1 if mutation_count > 10 else 0
            
            labels.append(label)
        
        labels = np.array(labels)
        logger.info(f"Created labels: {np.bincount(labels) if len(labels) > 0 else 'No labels'}")
        return labels
    
    def create_pytorch_geometric_data(self) -> Data:
        """
        Create PyTorch Geometric Data object with all multi-modal features
        """
        logger.info("Creating PyTorch Geometric Data object")
        
        # Load all data
        self.load_all_data()
        
        # Create features and get patient list
        features, patients = self.create_multi_modal_features()
        
        if len(patients) == 0:
            logger.error("No patients found")
            return None
        
        # Create advanced graph
        edge_index, edge_weights, edge_types = self.create_advanced_graph(patients)
        
        # Create labels
        labels = self.create_labels(patients)
        
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
        
        logger.info(f"Created PyTorch Geometric Data:")
        logger.info(f"  Nodes: {data.num_nodes}")
        logger.info(f"  Edges: {data.edge_index.shape[1]}")
        logger.info(f"  Features: {data.x.shape[1]}")
        logger.info(f"  Classes: {data.num_classes}")
        logger.info(f"  Edge attributes: {data.edge_attr.shape[1] if len(data.edge_attr) > 0 else 0}")
        
        return data
    
    def save_enhanced_data(self, data: Data) -> str:
        """
        Save enhanced multi-modal data
        """
        if data is None:
            logger.error("No data to save")
            return ""
        
        output_file = self.output_dir / "enhanced_multi_modal_data.pt"
        torch.save(data, output_file)
        
        # Save metadata
        metadata = {
            'num_nodes': data.num_nodes,
            'num_classes': data.num_classes,
            'feature_dim': data.x.shape[1],
            'edge_types': list(self.edge_types.keys()),
            'patient_ids': data.patient_ids
        }
        
        metadata_file = self.output_dir / "enhanced_multi_modal_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved enhanced multi-modal data to {output_file}")
        logger.info(f"Saved metadata to {metadata_file}")
        
        return str(output_file)

def main():
    """Main function to create enhanced multi-modal data"""
    logger.info("Starting enhanced multi-modal data processing")
    
    processor = EnhancedMultiModalProcessor()
    
    # Create PyTorch Geometric Data object
    data = processor.create_pytorch_geometric_data()
    
    if data is not None:
        # Save the data
        output_file = processor.save_enhanced_data(data)
        
        logger.info("Enhanced multi-modal data processing completed successfully")
        logger.info(f"Output file: {output_file}")
    else:
        logger.error("Failed to create enhanced multi-modal data")

if __name__ == "__main__":
    main() 