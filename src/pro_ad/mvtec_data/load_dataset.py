import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..logging import get_logger

logger = get_logger("MVTec-Ad Dataset Loader", logging.INFO)

class MVTecADJSONDataset(Dataset):
    """
    Load MvTec AD dataset from a JSON file.

    Supports the JSON format with entries like:
    {"filename": "zipper/train/good/002.png", "label": 0, "label_name": "good", "clsname": "zipper"}
    """
    def __init__(self, json_file: str | Path, root_dir: str|Path, category: Optional[str] = None, split:str = "train", is_normal: bool = True, img_size: int=224, augment: bool=False, load_masks: bool=False, return_path: bool = False,):
        """
        Args:
            json_file (str | Path): Path to the JSON file containing dataset metadata.
            root_dir (str | Path): Root directory where images are stored.
            category (Optional[str]): Specific category to filter by, e.g., "zipper". If None, all categories are included.
            split (str): Dataset split, either "train" or "test".
            is_normal (bool): If True, only load normal samples. If False, load all samples.
            img_size (int): Size to which images will be resized.
            augment (bool): Whether to apply data augmentation.
            load_masks (bool): Whether to load segmentation masks.
            return_path (bool): If True, return the image path in the sample.
        """
        self.root_dir = Path(root_dir)
        self.json_file = Path(json_file)
        self.category = category
        self.split = split
        self.is_normal = is_normal
        self.img_size = img_size
        self.load_masks = load_masks
        self.return_path = return_path

        # Load data from JSON file
        self.img_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        self.categories = []
        
        self._load_data_from_json()

        # Setup transform
        self.transform = self._get_transform(augment)
        self.mask_transform = (
            transforms.Compose(
                [
                    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ]
            )
            if load_masks
            else None
        )

        logger.info(
            "Loaded %d images from JSON for %s/%s%s",
            len(self.img_paths),
            category or "all_categories",
            split,
            "_normal" if is_normal else "_anomaly" if split == "test" else "",
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load images
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        category = self.categories[idx]

        res = {
            "image": img,
            "label": label,
            "defect_type": defect_type,
            "category": category,
        }

        # Load mask if required
        if (
            self.load_masks and idx<len(self.mask_paths) and self.mask_paths[idx] is not None
        ):
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            res["mask"] = mask
        else:
            res["mask"] = torch.zeros(1, self.img_size, self.img_size)

        # Add path if requested
        if self.return_path:
            res["path"] = img_path

        return res

    def _load_data_from_json(self):
        """
        Load data from JSON file and filter based on category, split, and is_normal.
        """
        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        
        with open(self.json_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {line}. Error: {e}")
                    continue
                
                filename = entry.get("filename", "")
                label = entry.get("label", 0)
                label_name = entry.get("label_name", "")
                clsname = entry.get("clsname", "")
                maskname = entry.get("maskname", "")
                
                # Filter by category if specified
                # if self.category is not None and clsname != self.category:
                #     continue
                
                # Parse filename to determine split and type
                path_parts = filename.split('/')
                if len(path_parts) < 3:
                    continue
                
                file_category = path_parts[0]
                file_split = path_parts[1]
                defect_type = path_parts[2] if len(path_parts) > 2 else ""
                
                # Filter by split
                if file_split != self.split:
                    continue
                
                # Filter by normal/anomaly for test split
                if self.split == "test":
                    is_sample_normal = (defect_type == "good" or label == 0)
                    if self.is_normal and not is_sample_normal:
                        continue
                    if not self.is_normal and is_sample_normal:
                        continue
                elif self.split == "train":
                    # Training data should typically be normal only
                    if label != 0 and defect_type != "good":
                        continue
                
                # Construct full image path
                img_path = self.root_dir / filename
                if not img_path.exists():
                    logger.warning(f"Image file not found: {img_path}")
                    continue
                
                # Add to lists
                self.img_paths.append(str(img_path))
                self.labels.append(label)
                self.defect_types.append(defect_type)
                self.categories.append(clsname)
                
                # For masks, use the maskname from JSON if available
                if self.load_masks and maskname:
                    mask_path = self.root_dir / maskname
                    if mask_path.exists():
                        self.mask_paths.append(str(mask_path))
                    else:
                        self.mask_paths.append(None)
                else:
                    self.mask_paths.append(None)

    def _get_transform(self, augment: bool):
        """
        Get the image transformation pipeline.
        """
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if augment and self.split == "train":
            # data augmentation for training
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

class MVTecADJSONDataModule:
    """
    Data module for MVTecAD dataset using JSON file format
    """
    def __init__(self, train_json: str, test_json: Optional[str] = None, root_dir: Optional[str] = None, category: Optional[str] = None, img_size: int = 224, batch_size: int = 32, num_workers: int=4, pin_memory: bool = True, augment_train: bool = True, load_masks: bool = False):
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
        if test_json is None:
            test_json = train_json

        self.train_json = train_json
        self.test_json = test_json
        self.root_dir = root_dir or str(Path(train_json).parent.parent)
        self.category = category
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.load_masks = load_masks

        self.train_dataset = MVTecADJSONDataset
        self.test_normal_dataset = MVTecADJSONDataset
        self.test_anomaly_dataset = MVTecADJSONDataset

        self._setup_datasets()

    def _setup_datasets(self):
        self.train_dataset = MVTecADJSONDataset(
            json_file=self.train_json,
            root_dir=self.root_dir,
            category = self.category,
            split="train",
            img_size=self.img_size,
            augment=self.augment_train,
            load_masks=False # No masks for normal training data
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

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_normal_loader(self) -> DataLoader:
        return DataLoader(
            self.test_normal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_anomaly_loader(self) -> DataLoader:
        return DataLoader(
            self.test_anomaly_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_combined_test_dataloader(self, shuffle: bool = False) -> DataLoader:
        """
        Get combine test dataloader (normal + anomaly)
        """
        combined_dataset = ConcatDataset([self.test_normal_dataset, self.test_anomaly_dataset])

        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_anomaly_feedback_dataloader(self, feedback_ratio: float=0.1, shuffle: bool=True) -> DataLoader:
        """
        Get a subset of anomaly data for feedback during training

        Args:
            feedback_ratio: Ratio of anomaly data to use as feedback
            shuffle: Whether to shuffle the data
        """
        dataset_size = len(self.test_anomaly_dataset)
        feedback_size = max(1, int(dataset_size * feedback_size))

        indices = torch.randperm(dataset_size)[:feedback_size]
        feedback_dataset = torch.utils.data.Subset(self.test_anomaly_dataset, indices.tolist())


        return DataLoader(feedback_dataset, batch_size-min(self.batch_size, feedback_size), shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

def main() -> None:
    # Example usage
    data_module = MVTecADJSONDataModule(
        train_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/train.json",
        test_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/test.json",
        root_dir="/home/yijin/projects/MemVim-AD/data/mvtec-ad",
        category="zipper",
        img_size=224,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        augment_train=True,
        load_masks=True,
    )

    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_combined_test_dataloader()

    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(f"Train batch - images: {images.shape}, labels: {labels.shape}")
        break

    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"]
        masks = batch.get("mask", None)
        print(f"Test batch - images: {images.shape}, labels: {labels.shape}, masks: {masks.shape if masks is not None else 'N/A'}")
        break

if __name__ == "__main__":
    main()