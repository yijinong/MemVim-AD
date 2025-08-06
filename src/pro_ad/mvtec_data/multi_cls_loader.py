import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..logging import get_logger
from .dataset import MVTecADDataModule, MVTecADDataset

logger = get_logger("Multi Category MVTecAD", logging.INFO)


class MultiCategoryMVTecAD:
    """
    Multi-category MVTecAD dataset manager for training across all categories
    """

    def __init__(
        self,
        root_dir: str,
        categories: Optional[List[str]] = None,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Args:
            root_dir: Path to MVTecAD dataset root
            categories: List of categories to include (None for all)
            img_size: Target image size
            batch_size: Batch size
            num_workers: Number of workers
        """
        self.root_dir = root_dir
        self.categories = categories or MVTecADDataset.CATEGORIES
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Data modules for each category
        self.data_modules = {}

        for category in self.categories:
            try:
                self.data_modules[category] = MVTecADDataModule(
                    root_dir=root_dir,
                    category=category,
                    img_size=img_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                logger.info(f"Successfully loaded category: {category}")
            except Exception as e:
                logger.warning(f"Failed to load category {category}: {e}")

    def get_category_datamodule(self, category: str) -> MVTecADDataModule:
        """Get data module for specific category"""
        if category not in self.data_modules:
            raise ValueError(f"Category {category} not available")
        return self.data_modules[category]

    def get_all_train_dataloaders(self) -> Dict[str, DataLoader]:
        """Get training dataloaders for all categories"""
        return {
            category: dm.get_train_dataloader()
            for category, dm in self.data_modules.items()
        }

    def print_dataset_info(self):
        """Print information about all loaded datasets"""
        print("\n" + "=" * 50)
        print("MVTecAD Dataset Information")
        print("=" * 50)

        total_train = 0
        total_test_normal = 0
        total_test_anomaly = 0

        for category, dm in self.data_modules.items():
            train_size = len(dm.train_dataset)
            test_normal_size = len(dm.test_normal_dataset)
            test_anomaly_size = len(dm.test_anomaly_dataset)

            print(
                f"{category:12} | Train: {train_size:4d} | "
                f"Test Normal: {test_normal_size:3d} | "
                f"Test Anomaly: {test_anomaly_size:3d}"
            )

            total_train += train_size
            total_test_normal += test_normal_size
            total_test_anomaly += test_anomaly_size

        print("-" * 50)
        print(
            f"{'Total:':12} | Train: {total_train:4d} | "
            f"Test Normal: {total_test_normal:3d} | "
            f"Test Anomaly: {total_test_anomaly:3d}"
        )
        print("=" * 50)


def collate_fn(batch):
    """Custom collate function for hierarchical prototype learning"""
    # Standard collation
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])

    # Additional information
    defect_types = [item["defect_type"] for item in batch]
    categories = [item["category"] for item in batch]

    result = {
        "image": images,
        "label": labels,
        "defect_type": defect_types,
        "category": categories,
    }

    # Add masks if present
    if "mask" in batch[0]:
        masks = torch.stack([item["mask"] for item in batch])
        result["mask"] = masks

    # Add paths if present
    if "path" in batch[0]:
        result["path"] = [item["path"] for item in batch]

    return result
