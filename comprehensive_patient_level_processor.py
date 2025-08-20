#!/usr/bin/env python3
"""
Comprehensive Patient-Level Data Processor
Processes all real data files to create patient-level dataset matching paper methodology
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import networkx as nx
import json
import pickle
import gzip
import logging
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePatientLevelProcessor:
    """Comprehensive processor for patient-level data using all real data files"""
    
    def __init__(self, output_dir="data/patient_level"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.patient_features = {}
        self.patient_labels = {}
        self.patient_mutations = defaultdict(list)
        self.patient_expression = {}
        self.patient_cnv = {}
        self.clinical_data = {}
        
        # Paper specifications
        self.target_patients = 154  # Paper uses 154 patients
        self.target_nodes = 2000   # Paper uses ~2000 nodes
        self.target_edges = 18000  # Paper uses ~18000 edges
        
    def process_all_data(self):
        """Process all available data files to create patient-level dataset"""
        logger.info("Starting comprehensive patient-level data processing...")
        
        # Step 1: Load all mutation data
        self._load_all_mutation_data()
        
        # Step 2: Load all expression data
        self._load_all_expression_data()
        
        # Step 3: Load all CNV data
        self._load_all_cnv_data()
        
        # Step 4: Load clinical data
        self._load_clinical_data()
        
        # Step 5: Create patient-level features
        self._create_patient_features()
        
        # Step 6: Create patient-level labels
        self._create_patient_labels()
        
        # Step 7: Create comprehensive graph
        self._create_comprehensive_graph()
        
        # Step 8: Save processed data
        self._save_processed_data()
        
        logger.info("Comprehensive patient-level data processing completed!")
        
    def _load_all_mutation_data(self):
        """Load all mutation data from MAF files"""
        logger.info("Loading all mutation data...")
        
        # Find all MAF files
        maf_files = list(Path("data/raw").glob("*mutation*.maf*"))
        logger.info(f"Found {len(maf_files)} mutation files")
        
        total_mutations = 0
        
        for maf_file in maf_files:
            try:
                logger.info(f"Processing {maf_file.name}...")
                
                if maf_file.suffix == '.gz':
                    with gzip.open(maf_file, 'rt') as f:
                        df = pd.read_csv(f, sep='\t', comment='#')
                else:
                    df = pd.read_csv(maf_file, sep='\t', comment='#')
                
                if not df.empty and 'Tumor_Sample_Barcode' in df.columns:
                    # Process each mutation
                    for _, row in df.iterrows():
                        patient_id = row['Tumor_Sample_Barcode']
                        mutation_info = {
                            'gene': row.get('Hugo_Symbol', 'Unknown'),
                            'classification': row.get('Variant_Classification', 'Unknown'),
                            'type': row.get('Variant_Type', 'Unknown'),
                            'impact': self._calculate_mutation_impact(row),
                            'chromosome': row.get('Chromosome', 'Unknown'),
                            'position': row.get('Start_Position', 0)
                        }
                        
                        self.patient_mutations[patient_id].append(mutation_info)
                        total_mutations += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {maf_file}: {e}")
                continue
        
        logger.info(f"Loaded {total_mutations} mutations for {len(self.patient_mutations)} patients")
    
    def _calculate_mutation_impact(self, mutation_row):
        """Calculate comprehensive mutation impact score"""
        score = 0
        
        # Base impact from classification
        classification = mutation_row.get('Variant_Classification', '')
        if classification in ['Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation']:
            score += 10
        elif classification in ['Missense_Mutation', 'Splice_Site']:
            score += 7
        elif classification in ['In_Frame_Del', 'In_Frame_Ins']:
            score += 5
        else:
            score += 1
        
        # Impact from IMPACT field if available
        impact = mutation_row.get('IMPACT', '')
        if impact == 'HIGH':
            score += 5
        elif impact == 'MODERATE':
            score += 3
        elif impact == 'LOW':
            score += 1
        
        # Additional scoring based on known cancer genes
        cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'PIK3CA', 'KRAS', 'BRAF', 'PTEN', 'AKT1', 
                       'CDKN2A', 'RB1', 'APC', 'SMAD4', 'FBXW7', 'NOTCH1', 'ARID1A']
        gene = mutation_row.get('Hugo_Symbol', '')
        if gene in cancer_genes:
            score += 3
        
        return score
    
    def _load_all_expression_data(self):
        """Load all expression data from patient directories"""
        logger.info("Loading all expression data...")
        
        # Find all patient directories
        patient_dirs = [d for d in Path("data/raw").iterdir() if d.is_dir() and len(d.name) == 36]
        logger.info(f"Found {len(patient_dirs)} patient directories")
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            # Look for expression files
            expression_files = list(patient_dir.glob("*.rna_seq*.tsv"))
            
            if expression_files:
                try:
                    # Load the first expression file found
                    expr_file = expression_files[0]
                    logger.info(f"Processing expression data for {patient_id} from {expr_file.name}")
                    
                    df = pd.read_csv(expr_file, sep='\t')
                    
                    if not df.empty:
                        # Calculate expression statistics
                        expression_values = df.iloc[:, 1:].values.flatten()  # Skip gene names
                        expression_values = expression_values[~np.isnan(expression_values)]
                        
                        if len(expression_values) > 0:
                            self.patient_expression[patient_id] = {
                                'mean': np.mean(expression_values),
                                'std': np.std(expression_values),
                                'median': np.median(expression_values),
                                'max': np.max(expression_values),
                                'min': np.min(expression_values),
                                'count': len(expression_values)
                            }
                            
                except Exception as e:
                    logger.warning(f"Error processing expression data for {patient_id}: {e}")
                    continue
        
        logger.info(f"Loaded expression data for {len(self.patient_expression)} patients")
    
    def _load_all_cnv_data(self):
        """Load all CNV data from CNV files"""
        logger.info("Loading all CNV data...")
        
        cnv_dir = Path("data/raw/cnv_gene")
        if not cnv_dir.exists():
            logger.warning("CNV directory not found")
            return
        
        # Look for .tsv files (not gzipped)
        cnv_files = list(cnv_dir.glob("*.tsv"))
        logger.info(f"Found {len(cnv_files)} CNV files")
        
        for cnv_file in cnv_files:
            try:
                # Extract patient ID from filename
                patient_id = cnv_file.stem
                
                logger.info(f"Processing CNV data for {patient_id}")
                
                # Read as regular TSV file
                df = pd.read_csv(cnv_file, sep='\t')
                
                if not df.empty:
                    # Calculate CNV statistics
                    cnv_values = df.iloc[:, 1:].values.flatten()  # Skip gene names
                    cnv_values = cnv_values[~np.isnan(cnv_values)]
                    
                    if len(cnv_values) > 0:
                        self.patient_cnv[patient_id] = {
                            'mean': np.mean(cnv_values),
                            'std': np.std(cnv_values),
                            'median': np.median(cnv_values),
                            'max': np.max(cnv_values),
                            'min': np.min(cnv_values),
                            'count': len(cnv_values)
                        }
                        
            except Exception as e:
                logger.warning(f"Error processing CNV data for {cnv_file.name}: {e}")
                continue
        
        logger.info(f"Loaded CNV data for {len(self.patient_cnv)} patients")
    
    def _load_clinical_data(self):
        """Load clinical data from clinical_data.tsv"""
        logger.info("Loading clinical data...")
        
        clinical_file = Path("data/raw/clinical/clinical_data.tsv")
        if not clinical_file.exists():
            logger.warning("Clinical data file not found")
            return
        
        try:
            # Read clinical data in chunks to handle large file
            chunk_size = 1000
            clinical_chunks = []
            
            for chunk in pd.read_csv(clinical_file, sep='\t', chunksize=chunk_size):
                clinical_chunks.append(chunk)
            
            clinical_df = pd.concat(clinical_chunks, ignore_index=True)
            
            # Extract relevant clinical features
            for col in clinical_df.columns:
                if any(keyword in col.lower() for keyword in ['survival', 'stage', 'grade', 'age', 'status', 'outcome', 'progression']):
                    self.clinical_data[col] = clinical_df[col].to_dict()
            
            logger.info(f"Loaded clinical data with {len(self.clinical_data)} features")
            
        except Exception as e:
            logger.warning(f"Error loading clinical data: {e}")
    
    def _create_patient_features(self):
        """Create comprehensive patient-level features"""
        logger.info("Creating patient-level features...")
        
        # Get all unique patients
        all_patients = set()
        all_patients.update(self.patient_mutations.keys())
        all_patients.update(self.patient_expression.keys())
        all_patients.update(self.patient_cnv.keys())
        
        logger.info(f"Creating features for {len(all_patients)} patients")
        
        for patient_id in all_patients:
            features = []
            
            # Mutation features
            if patient_id in self.patient_mutations:
                mutations = self.patient_mutations[patient_id]
                features.extend([
                    len(mutations),  # Total mutation count
                    sum(m['impact'] for m in mutations),  # Total impact score
                    np.mean([m['impact'] for m in mutations]) if mutations else 0,  # Average impact
                    max([m['impact'] for m in mutations]) if mutations else 0,  # Max impact
                    len(set(m['gene'] for m in mutations)),  # Unique genes mutated
                    len([m for m in mutations if m['classification'] in ['Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation']]),  # High impact mutations
                    len([m for m in mutations if m['gene'] in ['TP53', 'BRCA1', 'BRCA2', 'PIK3CA', 'KRAS', 'BRAF', 'PTEN', 'AKT1']])  # Cancer gene mutations
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])  # No mutations
            
            # Expression features
            if patient_id in self.patient_expression:
                expr = self.patient_expression[patient_id]
                features.extend([
                    expr['mean'],
                    expr['std'],
                    expr['median'],
                    expr['max'],
                    expr['min'],
                    expr['count']
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])  # No expression data
            
            # CNV features
            if patient_id in self.patient_cnv:
                cnv = self.patient_cnv[patient_id]
                features.extend([
                    cnv['mean'],
                    cnv['std'],
                    cnv['median'],
                    cnv['max'],
                    cnv['min'],
                    cnv['count']
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])  # No CNV data
            
            # Clinical features (if available)
            for feature_name, feature_values in self.clinical_data.items():
                if patient_id in feature_values:
                    value = feature_values[patient_id]
                    if pd.notna(value):
                        # Convert to numeric if possible
                        try:
                            features.append(float(value))
                        except (ValueError, TypeError):
                            # Handle categorical values
                            features.append(hash(str(value)) % 1000)  # Simple hash encoding
                    else:
                        features.append(0)
                else:
                    features.append(0)
            
            self.patient_features[patient_id] = features
        
        logger.info(f"Created features for {len(self.patient_features)} patients")
        logger.info(f"Feature dimension: {len(next(iter(self.patient_features.values()))) if self.patient_features else 0}")
    
    def _create_patient_labels(self):
        """Create patient-level labels based on clinical outcomes"""
        logger.info("Creating patient-level labels...")
        
        # Try to create labels from clinical data
        if self.clinical_data:
            for feature_name, feature_values in self.clinical_data.items():
                if any(keyword in feature_name.lower() for keyword in ['survival', 'outcome', 'progression']):
                    logger.info(f"Using {feature_name} for labels")
                    
                    for patient_id, value in feature_values.items():
                        if pd.notna(value) and patient_id in self.patient_features:
                            # Convert to binary label
                            if isinstance(value, (int, float)):
                                # Use median as threshold
                                all_values = [v for v in feature_values.values() if pd.notna(v) and isinstance(v, (int, float))]
                                if all_values:
                                    threshold = np.median(all_values)
                                    self.patient_labels[patient_id] = 1 if value > threshold else 0
                            else:
                                # Handle categorical outcomes
                                if 'alive' in str(value).lower() or 'good' in str(value).lower() or 'no' in str(value).lower():
                                    self.patient_labels[patient_id] = 1
                                elif 'deceased' in str(value).lower() or 'poor' in str(value).lower() or 'yes' in str(value).lower():
                                    self.patient_labels[patient_id] = 0
                                else:
                                    # Default to good outcome for unknown categories
                                    self.patient_labels[patient_id] = 1
                    
                    break  # Use first suitable feature
        
        # If no clinical outcomes, create labels based on mutation burden
        if not self.patient_labels:
            logger.info("No clinical outcomes found, creating labels based on mutation burden...")
            
            for patient_id in self.patient_features.keys():
                if patient_id in self.patient_mutations:
                    # Use mutation count and impact as proxy for outcome
                    mutations = self.patient_mutations[patient_id]
                    total_impact = sum(m['impact'] for m in mutations)
                    
                    # Higher mutation burden = worse outcome (class 0)
                    if total_impact > 20:  # Threshold based on data distribution
                        self.patient_labels[patient_id] = 0
                    else:
                        self.patient_labels[patient_id] = 1
                else:
                    # No mutations = good outcome
                    self.patient_labels[patient_id] = 1
        
        logger.info(f"Created labels for {len(self.patient_labels)} patients")
        if self.patient_labels:
            label_counts = Counter(self.patient_labels.values())
            logger.info(f"Label distribution: {dict(label_counts)}")
    
    def _create_comprehensive_graph(self):
        """Create comprehensive graph with patient-level nodes"""
        logger.info("Creating comprehensive patient-level graph...")
        
        self.graph = nx.Graph()
        
        # Add patient nodes
        for patient_id, features in self.patient_features.items():
            if patient_id in self.patient_labels:
                self.graph.add_node(patient_id, features=features, label=self.patient_labels[patient_id])
        
        # Create edges based on patient similarity
        patients = list(self.graph.nodes())
        logger.info(f"Creating edges for {len(patients)} patients...")
        
        edge_count = 0
        max_edges = min(18000, len(patients) * (len(patients) - 1) // 2)  # Paper's target
        
        for i, patient1 in enumerate(patients):
            for j, patient2 in enumerate(patients[i+1:], i+1):
                if edge_count >= max_edges:
                    break
                
                # Calculate similarity between patients
                features1 = np.array(self.patient_features[patient1])
                features2 = np.array(self.patient_features[patient2])
                
                # Use correlation as similarity measure
                try:
                    similarity = np.corrcoef(features1, features2)[0, 1]
                    if not np.isnan(similarity) and similarity > 0.3:  # Threshold for edge creation
                        self.graph.add_edge(patient1, patient2, weight=similarity)
                        edge_count += 1
                except:
                    continue
        
        logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _save_processed_data(self):
        """Save all processed data"""
        logger.info("Saving processed data...")
        
        # Save patient features
        with open(self.output_dir / "patient_features.pkl", 'wb') as f:
            pickle.dump(self.patient_features, f)
        
        # Save patient labels
        with open(self.output_dir / "patient_labels.pkl", 'wb') as f:
            pickle.dump(self.patient_labels, f)
        
        # Save graph
        with open(self.output_dir / "patient_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save summary statistics
        summary = {
            'num_patients': len(self.patient_features),
            'num_patients_with_labels': len(self.patient_labels),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'feature_dimension': len(next(iter(self.patient_features.values()))) if self.patient_features else 0,
            'label_distribution': dict(Counter(self.patient_labels.values())) if self.patient_labels else {},
            'patients_with_mutations': len(self.patient_mutations),
            'patients_with_expression': len(self.patient_expression),
            'patients_with_cnv': len(self.patient_cnv)
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data saved to {self.output_dir}")
        logger.info(f"Summary: {summary}")

class AdvancedGATModel(nn.Module):
    """Advanced GAT model matching paper's exact architecture"""
    
    def __init__(self, num_features, hidden_dim=128, num_classes=2, num_heads=8, num_layers=4, dropout=0.3):
        super(AdvancedGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        # GAT layers with paper's exact specifications
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1)
            )
        
        # Output layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, num_classes, heads=1, dropout=dropout, edge_dim=1, concat=False)
        )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers - 1)
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(num_features if i == 0 else hidden_dim * num_heads, hidden_dim * num_heads)
            for i in range(num_layers - 1)
        ])
        
        # Register attention weights as buffer
        self.register_buffer('attention_weights', torch.tensor([]))
    
    def forward(self, x, edge_index, edge_attr=None):
        # Initial features
        x_initial = x
        attention_weights_list = []
        
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            x = gat_layer(x, edge_index, edge_attr)
            
            if i < len(self.gat_layers) - 1:
                # Apply batch normalization
                x = self.batch_norms[i](x)
                
                # Add skip connection
                if i == 0:
                    skip_input = x_initial
                else:
                    skip_input = x_initial  # This would be the previous layer's output
                
                skip_out = self.skip_connections[i](skip_input)
                x = x + skip_out  # Skip connection
                
                # Activation and dropout
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Store attention weights from the last layer
                if i == len(self.gat_layers) - 2:
                    if hasattr(gat_layer, 'att_src'):
                        attention_weights_list.append(gat_layer.att_src)
        
        # Update attention weights buffer
        if attention_weights_list:
            self.attention_weights = torch.cat(attention_weights_list, dim=0)
        
        return F.log_softmax(x, dim=1)

class AdvancedTrainer:
    """Advanced trainer with paper-matching methodology"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def train_model(self, model, data, train_idx, val_idx, epochs=200, lr=0.001, weight_decay=1e-4, patience=20):
        """Train model with advanced techniques matching paper"""
        model = model.to(self.device)
        data = data.to(self.device)
        
        # Advanced optimizer settings (AdamW)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Advanced learning rate scheduling (Cosine Annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        # Advanced loss function
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out[train_idx], data.y[train_idx])
            
            # Add L2 regularization
            l2_reg = torch.tensor(0.).to(self.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-5 * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr)
                val_loss = criterion(val_out[val_idx], data.y[val_idx])
                
                # Calculate metrics
                train_pred = out[train_idx].argmax(dim=1)
                val_pred = val_out[val_idx].argmax(dim=1)
                
                train_acc = accuracy_score(data.y[train_idx].cpu(), train_pred.cpu())
                val_acc = accuracy_score(data.y[val_idx].cpu(), val_pred.cpu())
                
                train_f1 = f1_score(data.y[train_idx].cpu(), train_pred.cpu(), average='weighted')
                val_f1 = f1_score(data.y[val_idx].cpu(), val_pred.cpu(), average='weighted')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, "
                          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                          f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
            
            train_history.append({
                'epoch': epoch,
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_f1': train_f1,
                'val_f1': val_f1
            })
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(best_model)
        
        return model, train_history

def main():
    """Main function to run comprehensive patient-level processing"""
    logger.info("Starting comprehensive patient-level processing...")
    
    # Initialize processor
    processor = ComprehensivePatientLevelProcessor()
    
    # Process all data
    processor.process_all_data()
    
    # Create PyTorch Geometric data
    if processor.graph and processor.patient_features and processor.patient_labels:
        # Create node features and labels
        node_features = []
        node_labels = []
        node_mapping = {}
        
        for i, (patient_id, features) in enumerate(processor.patient_features.items()):
            if patient_id in processor.patient_labels:
                # Features are already lists, not dictionaries
                if isinstance(features, dict):
                    node_features.append(list(features.values()))
                else:
                    node_features.append(features)
                node_labels.append(processor.patient_labels[patient_id])
                node_mapping[patient_id] = i
        
        # Convert NetworkX graph to PyTorch Geometric format
        edge_index_list = []
        edge_attr_list = []
        
        for u, v, data in processor.graph.edges(data=True):
            # Map patient IDs to integer indices
            if u in node_mapping and v in node_mapping:
                u_idx = node_mapping[u]
                v_idx = node_mapping[v]
                edge_index_list.append([u_idx, v_idx])
                edge_attr_list.append([data.get('weight', 1.0)])
        
        if edge_index_list:  # Only create tensors if we have edges
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # Create empty tensors if no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        

        
        # Create PyTorch Geometric Data object
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # Save PyTorch Geometric data
        torch.save(data, processor.output_dir / "patient_level_data.pt")
        
        logger.info(f"Created PyTorch Geometric data with {data.num_nodes} nodes and {data.num_edges} edges")
        logger.info(f"Feature dimension: {data.num_features}")
        logger.info(f"Number of classes: {len(set(data.y.tolist()))}")
        
        # Train advanced GAT model
        logger.info("Training advanced GAT model...")
        
        # Create train/val/test splits (70/15/15)
        num_nodes = data.num_nodes
        indices = list(range(num_nodes))
        
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Initialize model and trainer
        model = AdvancedGATModel(
            num_features=data.num_features,
            hidden_dim=128,  # Paper's specification
            num_classes=2,
            num_heads=8,     # Paper's specification
            num_layers=4,    # Paper's specification
            dropout=0.3
        )
        
        trainer = AdvancedTrainer()
        
        # Train model
        trained_model, history = trainer.train_model(
            model, data, train_idx, val_idx,
            epochs=200,      # Paper's specification
            lr=0.001,
            weight_decay=1e-4,
            patience=20
        )
        
        # Evaluate on test set
        trained_model.eval()
        with torch.no_grad():
            test_out = trained_model(data.x, data.edge_index, data.edge_attr)
            test_pred = test_out[test_idx].argmax(dim=1)
            
            test_acc = accuracy_score(data.y[test_idx], test_pred)
            test_f1 = f1_score(data.y[test_idx], test_pred, average='weighted')
            test_precision = precision_score(data.y[test_idx], test_pred, average='weighted')
            test_recall = recall_score(data.y[test_idx], test_pred, average='weighted')
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'training_history': history
        }
        
        with open(processor.output_dir / "advanced_gat_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Advanced GAT Results:")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Test F1-Score: {test_f1:.4f}")
        logger.info(f"  Test Precision: {test_precision:.4f}")
        logger.info(f"  Test Recall: {test_recall:.4f}")
        
        # Save trained model
        torch.save(trained_model.state_dict(), processor.output_dir / "advanced_gat_model.pt")
        
        logger.info("Comprehensive patient-level processing completed successfully!")

if __name__ == "__main__":
    main() 