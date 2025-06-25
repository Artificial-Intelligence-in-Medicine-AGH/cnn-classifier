import os

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset

from augmentation.transform import train_transform, val_transform, test_transform
from config import config

# TODO: ----- Do osobnego csv -----
REF_CLASSES = [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Thickening",
    "Pneumonia",
    "Fibrosis",
    "Edema",
    "Consolidation",
]


class Case(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # This is needed to make pydantic validate torch.Tensors

    path: str
    labels: torch.Tensor


def load_data_from_csv(subfolder_name: str, labels_dict: dict[str, str]) -> list[Case]:
    folder_path = os.path.join(config.dataset_path, subfolder_name)

    dataset: list[Case] = []

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            human_redable_labels = labels_dict[file].split('|')
            labels = torch.zeros(14)
            # Petla ktora generuje wlasciwy format, czyli nazwa pliku i lista chorob z typem int
            # Ze zrodla, czyli Hernia|Infiltration dostajemy to: labels=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Pozycja choroby okreslona jest w zahardcodowanej tablicy ref_classes
            for single_label in human_redable_labels:
                if single_label in REF_CLASSES:
                    idx = REF_CLASSES.index(single_label)
                    labels[idx] = 1
                elif single_label != 'No Finding':
                    print(f"Label not found: {single_label}")
            file_path = os.path.join(folder_path, file)
            train_case = Case(path=file_path, labels=labels)
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
    labelsTable = np.genfromtxt(config.labels_file_path, delimiter=',', dtype=str, usecols=(0, 1), skip_header=1)

    # Kluczem jest nazwa pliku, wartoscia sa nazwy klas, np. 00000001_000.png -> Hernia|Mass|Nodule
    labels_dict: dict[str, str] = dict(labelsTable)  # Labels table to lista par (klucz -> wartosc).

    train_data = load_data_from_csv("train", labels_dict)
    train_dataset = CustomDataset(train_data, transform=train_transform)

    val_data = load_data_from_csv("val", labels_dict)
    val_dataset = CustomDataset(val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def get_test_loader():
    labelsTable = np.genfromtxt(
        config.labels_file_path, delimiter=",", dtype=str, usecols=(0, 1), skip_header=1
    )
    labels_dict: dict[str, str] = dict(labelsTable)
    test_data = load_data_from_csv("test", labels_dict)
    test_dataset = CustomDataset(test_data, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False, num_workers=4)

    return test_loader
