import os

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from augmentation.transform import train_transform, val_transform
from config import config

# TODO: ----- Do osobnego csv -----
REF_CLASSES = config.classes


class Case(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # This is needed to make pydantic validate torch.Tensors

    path: str
    labels: torch.Tensor


def load_data_from_csv(subfolder_name: str, labels_dict: dict[str, str]) -> list[Case]:
    
    folder_path = os.path.join(config.dataset_path, subfolder_name)
    dataset: list[Case] = []

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            lables = torch.tensor([float(n) for n in labels_dict[file]])
            train_case = Case(path=file_path, labels=lables)
            dataset.append(train_case)

    return dataset


class CustomDataset(Dataset):
    def __init__(self, data: list[Case], transform):
        self.data: list[Case] = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: Case = self.data[idx]
        img = Image.open(item.path)
        img = np.array(img)
        augmentations_result = self.transform(image=img)
        img = augmentations_result["image"]

        label = item.labels
        return img, label


def get_loaders() -> tuple[DataLoader, DataLoader]:
    labelsTable:pd.DataFrame = pd.read_csv(os.path.join(config.dataset_path, "bit_map_data.csv"), dtype=str)
    labels_dict = dict(zip([row[1] for row in labelsTable.itertuples()],[row[2] for row in labelsTable.itertuples()]))

    train_data = load_data_from_csv("train", labels_dict)
    train_dataset = CustomDataset(train_data, transform=train_transform)

    val_data = load_data_from_csv("val", labels_dict)
    val_dataset = CustomDataset(val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
