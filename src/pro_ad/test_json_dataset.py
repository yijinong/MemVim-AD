#!/usr/bin/env python3
"""
Test script for the JSON-based MVTecAD dataset
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from .mvtec_data.load_dataset import MVTecADJSONDataModule, MVTecADJSONDataset


def test_json_dataset():
    """Test the JSON dataset implementation"""

    # Paths
    train_json = "/home/yijin/projects/MemVim-AD/data/mvtec-ad/train.json"
    test_json = "/home/yijin/projects/MemVim-AD/data/mvtec-ad/test.json"
    data_root = "/home/yijin/projects/MemVim-AD/data/mvtec-ad"

    print("Testing MVTecADJSONDataset...")

    # Test loading zipper training data
    train_dataset = MVTecADJSONDataset(
        json_file=train_json,
        root_dir=data_root,
        category="zipper",
        split="train",
        img_size=224,
        return_path=True,
    )

    print(f"Training dataset size: {len(train_dataset)}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Defect type: {sample['defect_type']}")
        print(f"Category: {sample['category']}")
        print(f"Path: {sample['path']}")

    # Test loading metal_nut training data
    train_dataset_metal = MVTecADJSONDataset(
        json_file=train_json,
        root_dir=data_root,
        category="metal_nut",
        split="train",
        img_size=224,
    )

    print(f"Metal_nut training dataset size: {len(train_dataset_metal)}")

    # Test data module
    print("\nTesting MVTecADJSONDataModule...")

    data_module = MVTecADJSONDataModule(
        train_json=train_json,
        test_json=test_json,
        root_dir=data_root,
        category="zipper",
        batch_size=4,
        img_size=224,
    )

    # Test dataloaders
    train_loader = data_module.get_train_loader()
    print(f"Train dataloader batch size: {train_loader.batch_size}")
    print(f"Train dataset size: {len(train_loader.dataset)}")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels: {batch['label']}")
    print(f"Batch categories: {batch['category']}")


if __name__ == "__main__":
    test_json_dataset()
