import pytest
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

from graph.graph_dataset import AgingDataset
from torch_geometric.data import Data

def test_aging_dataset():
    # Setup dummy directory
    root = Path("./temp_test_data")
    
    cfg = OmegaConf.create({
        'training': {
            'seq_len': 5
        }
    })
    
    # 1. Init empty
    dataset = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset) == 0
    
    # 2. Add dummy samples
    for i in range(3):
        # 10 nodes, 5 features, sequence targets
        d = Data(
            x=torch.rand(10, 5),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.rand(10, 1),
            y_trajectory=torch.rand(10, 5)
        )
        dataset.add_sample(d)
        
    # 3. Save
    dataset.finalize_and_save()
    
    # 4. Load
    dataset2 = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset2) == 3
    
    # Access
    sample = dataset2[0]
    assert sample.x.shape == (10, 5)
    
    # Cleanup
    import shutil
    shutil.rmtree(root, ignore_errors=True)
