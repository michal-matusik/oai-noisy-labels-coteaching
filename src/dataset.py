import os
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.folder import VisionDataset
import torchvision.transforms as transforms

IMAGES_DIR = "data"
TASK_DATASET_LABELS_FILE = "dataset_labels.csv"


def download_data(dataset_path: str, dataset_url: str) -> None:
    """Downloads a dataset archive from Google Drive (competition-provided)."""
    import gdown
    import shutil

    output = dataset_path + ".zip"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    if os.path.exists(output):
        os.remove(output)
    gdown.download(id=dataset_url, output=output)
    print(f"Downloaded: {output}")


def unpack_data(unpack_path: str, dataset_name: str) -> None:
    dataset_zip_path = os.path.join(unpack_path, dataset_name + ".zip")
    dataset_local_dir = os.path.join(unpack_path, dataset_name)
    if not os.path.exists(dataset_local_dir):
        if not os.path.exists(dataset_zip_path):
            raise FileNotFoundError(f"{dataset_zip_path} not found.")
        with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
            zip_ref.extractall(unpack_path)


class TaskDataset(VisionDataset):
    """Binary-classification image dataset with (possibly noisy) labels in a CSV file.

    Competition-provided; do not modify.
    """

    def __init__(self, root: str, transform: Optional[callable] = None):
        super().__init__(root, transform=transform)
        self.root = root
        if not self._check_integrity():
            raise RuntimeError(
                f"Dataset not found. Expected '{IMAGES_DIR}/' and "
                f"'{TASK_DATASET_LABELS_FILE}' under {self.root}"
            )
        self.labels_df = self._read_labels_from_file()
        self.labels_header = "label"

    def _read_labels_from_file(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.root, TASK_DATASET_LABELS_FILE))

    def _check_integrity(self) -> bool:
        return os.path.exists(os.path.join(self.root, IMAGES_DIR)) and os.path.exists(
            os.path.join(self.root, TASK_DATASET_LABELS_FILE)
        )

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        img = self._load_image(idx)
        label = self._load_label(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _load_image(self, idx: int) -> Image.Image:
        img_path = os.path.join(self.root, IMAGES_DIR, self.labels_df.iloc[idx]["file_name"])
        return Image.open(img_path)

    def _load_label(self, idx: int):
        return np.array([int(self.labels_df.iloc[idx][self.labels_header])])


def load_data(train_path: str, val_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    base_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TaskDataset(root=train_path, transform=base_transform)
    val_dataset = TaskDataset(root=val_path, transform=base_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
