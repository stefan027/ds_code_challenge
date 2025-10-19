"""Data related classes and functions for the image classification task."""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TiffImageDataset(Dataset):
    """
    PyTorch Dataset for reading .tif images and preparing them for image classification.
    """

    def __init__(self, paths, labels=None, transform=None, use_albumentations=False):
        """
        Args:
            paths (list): List of file paths to .tif images.
            labels (list, optional): List of integer labels corresponding to each image. 
                                     If None, dataset returns only images.
            transform (callable, optional): Optional transform to be applied to each image.
        """
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_albumentations = use_albumentations

        if self.use_albumentations:
            # ToTensorV2 should be part of the transform pipeline
            if not any(isinstance(t, ToTensorV2) for t in self.transform.transforms):
                self.transform = A.Compose(self.transform.transforms + [ToTensorV2()])

        if self.transform is None:
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.use_albumentations = False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.use_albumentations:
            image = np.array(image)
            image = self.transform(image=image)["image"].to(self.device)
        else:
            image = self.transform(image).to(self.device)

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float, device=self.device)
        else:
            return image
