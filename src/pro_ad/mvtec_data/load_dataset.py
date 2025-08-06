import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils,data import ConcatDataset, DataLoader, Dataset

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
            len(self.image_paths),
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

    def _get_transform(self, augment: bool):
        """
        Get the image transformation pipeline.
        """
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