"""
Test script to verify GAT optimization is working
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gat_model():
    """Test the optimized GAT model"""
    logger.info("Testing optimized GAT model...")
    
    try:
        # Import our optimized GAT model
        from optimized_gat_implementation import OptimizedGATModel, AdvancedTrainingConfig
        
        # Create test data
        num_nodes = 50
        num_features = 20
        num_edges = 100
        
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randint(0, 8, (num_edges, 1)).float()
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Create model
        config = AdvancedTrainingConfig()
        model = OptimizedGATModel(
            input_dim=num_features,
            hidden_dim=config.hidden_dim,
            output_dim=2,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_edge_attr=True,
            num_edge_types=8,
            use_skip_connections=True,
            use_graph_attention=True,
            pooling_strategy='multi'
        )
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_attr, batch)
            logger.info(f"✓ Model output shape: {output.shape}")
            logger.info(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test training step
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy labels
        labels = torch.randint(0, 2, (num_nodes,))
        
        # Forward pass
        output = model(x, edge_index, edge_attr, batch)
        loss = criterion(output, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ Training step completed successfully")
        logger.info(f"✓ Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing GAT model: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    logger.info("Testing data loading...")
    
    try:
        data_path = Path("data/enhanced/real_only_torch_geometric_data.pt")
        if data_path.exists():
            data = torch.load(data_path, weights_only=False)
            logger.info(f"✓ Data loaded successfully")
            logger.info(f"✓ Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
            logger.info(f"✓ Edges: {data.edge_index.shape[1]}")
            if hasattr(data, 'y') and data.y is not None:
                logger.info(f"✓ Labels: {data.y.shape}")
            return True
        else:
            logger.warning("⚠ Data file not found")
            return False
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        return False

def main():
    """Main test function"""
    logger.info("="*50)
    logger.info("GAT OPTIMIZATION TEST")
    logger.info("="*50)
    
    # Test data loading
    data_success = test_data_loading()
    
    # Test GAT model
    model_success = test_gat_model()
    
    # Summary
    logger.info("="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Data loading: {'✓ PASS' if data_success else '✗ FAIL'}")
    logger.info(f"GAT model: {'✓ PASS' if model_success else '✗ FAIL'}")
    
    if data_success and model_success:
        logger.info("🎉 All tests passed! GAT optimization is working correctly.")
    else:
        logger.warning("⚠ Some tests failed. Check the logs above.")
    
    logger.info("="*50)

if __name__ == "__main__":
    main()
