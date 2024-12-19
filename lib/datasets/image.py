from typing import Dict, List
from PIL import Image
import torchvision
import torch.utils.data as td


class ImageDataset(td.Dataset):
    def __init__(self, image_paths: List, labels: List, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of corresponding labels for the images.
        """
        self.image_paths = image_paths
        self.labels = labels

        self.transform = transform or torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),  # Convert image to PyTorch tensor
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return (image, label)
