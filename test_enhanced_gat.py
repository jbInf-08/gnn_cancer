"""
Quick test for enhanced GAT model
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_gat():
    """Test the enhanced GAT model"""
    logger.info("Testing enhanced GAT model...")
    
    try:
        # Import our enhanced GAT model
        from enhanced_gat_v2 import EnhancedGATv2Model, AdvancedTrainingConfig
        
        # Create test data
        num_nodes = 50
        num_features = 19  # Match our actual data
        num_edges = 100
        
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randint(0, 8, (num_edges, 1)).float()  # 8 edge types
        
        # Create model
        model = EnhancedGATv2Model(
            input_dim=num_features,
            hidden_dim=128,
            output_dim=2,
            num_layers=3,
            num_heads=4,
            dropout=0.3,
            use_edge_attr=True,
            num_edge_types=8,
            use_skip_connections=True,
            use_multi_scale=True,
            use_attention_pooling=True,
            use_layer_norm=True,
            pooling_strategy='multi'
        )
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_attr)
            logger.info(f"✓ Forward pass successful")
            logger.info(f"  Input shape: {x.shape}")
            logger.info(f"  Output shape: {output.shape}")
            logger.info(f"  Expected output shape: ({num_nodes}, 2)")
            
            # Test with batch (graph-level prediction)
            batch = torch.zeros(num_nodes, dtype=torch.long)
            batch[num_nodes//2:] = 1  # Create 2 graphs
            
            output_batch = model(x, edge_index, edge_attr, batch)
            logger.info(f"✓ Batch forward pass successful")
            logger.info(f"  Batch output shape: {output_batch.shape}")
            logger.info(f"  Expected batch output shape: (2, 2)")
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    logger.info("Testing data loading...")
    
    try:
        data_file = Path("data") / "enhanced" / "real_only_torch_geometric_data.pt"
        if data_file.exists():
            data = torch.load(data_file, weights_only=False)
            logger.info(f"✓ Data loading successful")
            logger.info(f"  Nodes: {data.x.shape[0]}")
            logger.info(f"  Features: {data.x.shape[1]}")
            logger.info(f"  Edges: {data.edge_index.shape[1]}")
            logger.info(f"  Edge features: {data.edge_attr.shape[1] if data.edge_attr is not None else 0}")
            return True
        else:
            logger.error(f"✗ Data file not found: {data_file}")
            return False
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Running enhanced GAT tests...")
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test enhanced GAT model
    model_ok = test_enhanced_gat()
    
    if data_ok and model_ok:
        logger.info("🎉 All tests passed! Enhanced GAT is ready for training.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
