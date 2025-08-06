import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..logging import get_logger

logger = get_logger("MVTec-AD dataset", logging.INFO)


class MVTecADDataset(Dataset):
    """
    MVTecAD Dataset for Hierarchical Prototype Learning

    Supports:
    - Normal/Anomaly data loading
    - Multiple categories
    - Data augmentation
    - Mask loading for anomaly localization
    """

    CATEGORIES = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
    ]

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",  # 'train', 'test'
        is_normal: bool = True,  # For test split: True=normal, False=anomaly
        img_size: int = 224,
        augment: bool = False,
        load_masks: bool = False,
        return_path: bool = False,
    ):
        """
        Args:
            root_dir: Path to MVTecAD dataset root
            category: Category name (e.g., 'bottle', 'cable')
            split: 'train' or 'test'
            is_normal: For test split, whether to load normal or anomaly samples
            img_size: Target image size
            augment: Whether to apply data augmentation
            load_masks: Whether to load ground truth masks (only for anomalies)
            return_path: Whether to return image paths
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.is_normal = is_normal
        self.img_size = img_size
        self.load_masks = load_masks
        self.return_path = return_path

        assert category in self.CATEGORIES, (
            f"Category {category} not in {self.CATEGORIES}"
        )
        assert split in ["train", "test"], (
            f"Split must be 'train' or 'test', got {split}"
        )

        # Build image paths
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        self._load_dataset()

        # Setup transforms
        self.transform = self._get_transforms(augment)
        self.mask_transform = (
            transforms.Compose(
                [
                    transforms.Resize(
                        (img_size, img_size),
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ),
                    transforms.ToTensor(),
                ]
            )
            if load_masks
            else None
        )

        logger.info(
            "Loaded %d images for %s/%s%s",
            len(self.image_paths),
            category,
            split,
            "_normal" if is_normal else "_anomaly" if split == "test" else "",
        )

    def _load_dataset(self):
        """Load dataset paths and labels"""
        category_dir = self.root_dir / self.category

        if self.split == "train":
            # Training data is always normal
            train_dir = category_dir / "train" / "good"
            if train_dir.exists():
                for img_path in sorted(train_dir.glob("*.png")):
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # Normal = 0
                    self.defect_types.append("good")
                    if self.load_masks:
                        self.mask_paths.append(None)  # No masks for normal images

        elif self.split == "test":
            test_dir = category_dir / "test"

            if self.is_normal:
                # Load test normal images
                normal_dir = test_dir / "good"
                if normal_dir.exists():
                    for img_path in sorted(normal_dir.glob("*.png")):
                        self.image_paths.append(str(img_path))
                        self.labels.append(0)  # Normal = 0
                        self.defect_types.append("good")
                        if self.load_masks:
                            self.mask_paths.append(None)
            else:
                # Load test anomaly images
                ground_truth_dir = category_dir / "ground_truth"

                for defect_dir in test_dir.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != "good":
                        defect_type = defect_dir.name

                        for img_path in sorted(defect_dir.glob("*.png")):
                            self.image_paths.append(str(img_path))
                            self.labels.append(1)  # Anomaly = 1
                            self.defect_types.append(defect_type)

                            # Find corresponding mask
                            if self.load_masks:
                                mask_path = (
                                    ground_truth_dir / defect_type / img_path.name
                                )
                                if mask_path.exists():
                                    self.mask_paths.append(str(mask_path))
                                else:
                                    self.mask_paths.append(None)

    def _get_transforms(self, augment: bool):
        """Get image transforms"""
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment and self.split == "train":
            # Data augmentation for training
            augment_transforms = [
                transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
            return transforms.Compose(augment_transforms)

        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        result = {
            "image": image,
            "label": label,
            "defect_type": defect_type,
            "category": self.category,
        }

        # Load mask if requested
        if (
            self.load_masks
            and idx < len(self.mask_paths)
            and self.mask_paths[idx] is not None
        ):
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result["mask"] = mask
        else:
            result["mask"] = torch.zeros(1, self.img_size, self.img_size)

        # Add path if requested
        if self.return_path:
            result["path"] = img_path

        return result


class MVTecADJSONDataset(Dataset):
    """
    MVTecAD Dataset that loads from JSON file format

    Supports the JSON format with entries like:
    {"filename": "zipper/train/good/002.png", "label": 0, "label_name": "good", "clsname": "zipper"}
    """

    def __init__(
        self,
        json_file: str,
        root_dir: str,
        category: Optional[str] = None,  # If None, loads all categories
        split: str = "train",  # 'train', 'test'
        is_normal: bool = True,  # For test split: True=normal, False=anomaly
        img_size: int = 224,
        augment: bool = False,
        load_masks: bool = False,
        return_path: bool = False,
    ):
        """
        Args:
            json_file: Path to JSON file containing dataset entries
            root_dir: Path to dataset root directory containing images
            category: Category name (e.g., 'bottle', 'zipper'). If None, loads all categories
            split: 'train' or 'test'
            is_normal: For test split, whether to load normal or anomaly samples
            img_size: Target image size
            augment: Whether to apply data augmentation
            load_masks: Whether to load ground truth masks (only for anomalies)
            return_path: Whether to return image paths
        """
        self.root_dir = Path(root_dir)
        self.json_file = json_file
        self.category = category
        self.split = split
        self.is_normal = is_normal
        self.img_size = img_size
        self.load_masks = load_masks
        self.return_path = return_path

        # Load data from JSON
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        self.categories = []

        self._load_from_json()

        # Setup transforms
        self.transform = self._get_transforms(augment)
        self.mask_transform = (
            transforms.Compose(
                [
                    transforms.Resize(
                        (img_size, img_size),
                        interpolation=transforms.InterpolationMode.NEAREST,
                    ),
                    transforms.ToTensor(),
                ]
            )
            if load_masks
            else None
        )

        logger.info(
            "Loaded %d images from JSON for %s/%s%s",
            len(self.image_paths),
            category or "all_categories",
            split,
            "_normal" if is_normal else "_anomaly" if split == "test" else "",
        )

    def _load_from_json(self):
        """Load dataset from JSON file"""
        with open(self.json_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())

                filename = entry["filename"]
                label = entry["label"]
                label_name = entry["label_name"]
                clsname = entry["clsname"]

                # Parse split from filename path
                path_parts = filename.split("/")
                file_split = path_parts[1]  # e.g., 'train' or 'test'

                # Filter by split
                if file_split != self.split:
                    continue

                # Filter by category if specified
                if self.category is not None and clsname != self.category:
                    continue

                # Filter by normal/anomaly for test split
                if self.split == "test":
                    is_sample_normal = label == 0
                    if is_sample_normal != self.is_normal:
                        continue
                elif self.split == "train":
                    # Training should only have normal samples
                    if label != 0:
                        continue

                # Build full image path
                img_path = self.root_dir / filename
                if not img_path.exists():
                    continue

                self.image_paths.append(str(img_path))
                self.labels.append(label)
                self.defect_types.append(label_name)
                self.categories.append(clsname)

                # Handle masks for anomalies
                if self.load_masks and label == 1:
                    # Construct mask path (assume ground_truth structure)
                    mask_filename = filename.replace("/test/", "/ground_truth/")
                    mask_path = self.root_dir / mask_filename
                    if mask_path.exists():
                        self.mask_paths.append(str(mask_path))
                    else:
                        self.mask_paths.append(None)
                else:
                    self.mask_paths.append(None)

    def _get_transforms(self, augment: bool):
        """Get image transforms"""
        base_transforms = [
            transforms.resize((self.img_size, self.img_size)),
            transforms.totensor(),
            transforms.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment and self.split == "train":
            # data augmentation for training
            augment_transforms = [
                transforms.resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.randomcrop((self.img_size, self.img_size)),
                transforms.randomhorizontalflip(p=0.5),
                transforms.randomrotation(degrees=15),
                transforms.colorjitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.totensor(),
                transforms.normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
            return transforms.compose(augment_transforms)

        return transforms.compose(base_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        category = self.categories[idx]

        result = {
            "image": image,
            "label": label,
            "defect_type": defect_type,
            "category": category,
        }

        # Load mask if requested
        if (
            self.load_masks
            and idx < len(self.mask_paths)
            and self.mask_paths[idx] is not None
        ):
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result["mask"] = mask
        else:
            result["mask"] = torch.zeros(1, self.img_size, self.img_size)

        # Add path if requested
        if self.return_path:
            result["path"] = img_path

        return result


class MVTecADJSONDataModule:
    """
    Data module for MVTecAD dataset using JSON file format
    """

    def __init__(
        self,
        train_json: str,
        test_json: Optional[str] = None,  # If None, uses train_json for both
        root_dir: Optional[str] = None,
        category: Optional[str] = None,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train: bool = True,
        load_masks: bool = False,
    ):
        """
        Args:
            train_json: Path to JSON file containing training data
            test_json: Path to JSON file containing test data (if None, uses train_json)
            root_dir: Path to dataset root directory containing images
            category: Category name (if None, loads all categories)
            img_size: Target image size
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            augment_train: Whether to augment training data
            load_masks: Whether to load ground truth masks
        """
        self.train_json = train_json
        self.test_json = test_json or train_json
        self.root_dir = root_dir or str(
            Path(train_json).parent.parent
        )  # Infer from JSON path
        self.category = category
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.load_masks = load_masks

        # Datasets
        self.train_dataset: MVTecADJSONDataset
        self.test_normal_dataset: MVTecADJSONDataset
        self.test_anomaly_dataset: MVTecADJSONDataset

        self._setup_datasets()

    def _setup_datasets(self):
        """Setup all datasets"""
        # Training dataset (normal only)
        self.train_dataset = MVTecADJSONDataset(
            json_file=self.train_json,
            root_dir=self.root_dir,
            category=self.category,
            split="train",
            img_size=self.img_size,
            augment=self.augment_train,
            load_masks=False,  # No masks for normal training data
        )

        # Test normal dataset
        self.test_normal_dataset = MVTecADJSONDataset(
            json_file=self.test_json,
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            is_normal=True,
            img_size=self.img_size,
            augment=False,
            load_masks=self.load_masks,
        )

        # Test anomaly dataset
        self.test_anomaly_dataset = MVTecADJSONDataset(
            json_file=self.test_json,
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            is_normal=False,
            img_size=self.img_size,
            augment=False,
            load_masks=self.load_masks,
        )

        logger.info("Setup JSON datasets for %s:", self.category or "all_categories")
        logger.info("  Train: %d samples", len(self.train_dataset))
        logger.info("  Test Normal: %d samples", len(self.test_normal_dataset))
        logger.info("  Test Anomaly: %d samples", len(self.test_anomaly_dataset))

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader (normal samples only)"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # For contrastive learning consistency
        )

    def get_test_normal_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test normal dataloader"""
        return DataLoader(
            self.test_normal_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_anomaly_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test anomaly dataloader (for feedback during training)"""
        return DataLoader(
            self.test_anomaly_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_combined_test_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get combined test dataloader (normal + anomaly)"""
        combined_dataset = ConcatDataset(
            [self.test_normal_dataset, self.test_anomaly_dataset]
        )

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_anomaly_feedback_dataloader(
        self, feedback_ratio: float = 0.1, shuffle: bool = True
    ) -> DataLoader:
        """
        Get a subset of anomaly data for feedback during training

        Args:
            feedback_ratio: Ratio of anomaly data to use as feedback
            shuffle: Whether to shuffle the data
        """
        # Create subset of anomaly data
        dataset_size = len(self.test_anomaly_dataset)
        feedback_size = max(1, int(dataset_size * feedback_ratio))

        indices = torch.randperm(dataset_size)[:feedback_size]
        feedback_dataset = torch.utils.data.Subset(
            self.test_anomaly_dataset, indices.tolist()
        )

        return DataLoader(
            feedback_dataset,
            batch_size=min(self.batch_size, feedback_size),
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class MVTecADDataModule:
    """
    Data module for MVTecAD dataset with hierarchical prototype learning support
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train: bool = True,
        load_masks: bool = False,
    ):
        """
        Args:
            root_dir: Path to MVTecAD dataset root
            category: Category name
            img_size: Target image size
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            augment_train: Whether to augment training data
            load_masks: Whether to load ground truth masks
        """
        self.root_dir = root_dir
        self.category = category
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.load_masks = load_masks

        # Datasets
        self.train_dataset: MVTecADDataset
        self.test_normal_dataset: MVTecADDataset
        self.test_anomaly_dataset: MVTecADDataset

        self._setup_datasets()

    def _setup_datasets(self):
        """Setup all datasets"""
        # Training dataset (normal only)
        self.train_dataset = MVTecADDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="train",
            img_size=self.img_size,
            augment=self.augment_train,
            load_masks=False,  # No masks for normal training data
        )

        # Test normal dataset
        self.test_normal_dataset = MVTecADDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            is_normal=True,
            img_size=self.img_size,
            augment=False,
            load_masks=self.load_masks,
        )

        # Test anomaly dataset
        self.test_anomaly_dataset = MVTecADDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            is_normal=False,
            img_size=self.img_size,
            augment=False,
            load_masks=self.load_masks,
        )

        logger.info("Setup datasets for %s:", self.category)
        logger.info("  Train: %d samples", len(self.train_dataset))
        logger.info("  Test Normal: %d samples", len(self.test_normal_dataset))
        logger.info("  Test Anomaly: %d samples", len(self.test_anomaly_dataset))

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader (normal samples only)"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # For contrastive learning consistency
        )

    def get_test_normal_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test normal dataloader"""
        return DataLoader(
            self.test_normal_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_anomaly_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test anomaly dataloader (for feedback during training)"""
        return DataLoader(
            self.test_anomaly_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_combined_test_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get combined test dataloader (normal + anomaly)"""

        combined_dataset = ConcatDataset(
            [self.test_normal_dataset, self.test_anomaly_dataset]
        )

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_anomaly_feedback_dataloader(
        self, feedback_ratio: float = 0.1, shuffle: bool = True
    ) -> DataLoader:
        """
        Get a subset of anomaly data for feedback during training

        Args:
            feedback_ratio: Ratio of anomaly data to use as feedback
            shuffle: Whether to shuffle the data
        """
        # Create subset of anomaly data
        dataset_size = len(self.test_anomaly_dataset)
        feedback_size = max(1, int(dataset_size * feedback_ratio))

        indices = torch.randperm(dataset_size)[:feedback_size]
        feedback_dataset = torch.utils.data.Subset(
            self.test_anomaly_dataset, indices.tolist()
        )

        return DataLoader(
            feedback_dataset,
            batch_size=min(self.batch_size, feedback_size),
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
