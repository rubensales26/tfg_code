import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from abc import ABC, abstractmethod


class DataModule(ABC):
    """The abstract contract for all datasets in your thesis."""
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    @abstractmethod
    def train_dataloader(self): pass

    @abstractmethod
    def val_dataloader(self): pass

    @abstractmethod
    def test_dataloader(self): pass


class NumpyDataset(Dataset):
    """A general-purpose dataset for loading .npz files."""
    def __init__(self, imgs_path, labels_path):
        self.imgs = np.load(imgs_path)['arr_0']
        self.labels = np.load(labels_path)['arr_0']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32) / 255.0
        # Check if we need to add a channel dimension
        if img.ndim == 2:
            img = img.unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class KMnistDataModule(DataModule):
    def __init__(self, batch_size=32, data_dir='./datasets/KMNIST/raw'):
        super().__init__(batch_size)
        
        train_ds = NumpyDataset(f"{data_dir}/kmnist-train-imgs.npz", f"{data_dir}/kmnist-train-labels.npz")
        test_ds = NumpyDataset(f"{data_dir}/kmnist-test-imgs.npz", f"{data_dir}/kmnist-test-labels.npz")
        
        self.train_set, self.val_set = random_split(
            train_ds, [50000, 10000], generator=torch.Generator().manual_seed(42)
        )
        self.test_set = test_ds

    def train_dataloader(self): return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self): return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)