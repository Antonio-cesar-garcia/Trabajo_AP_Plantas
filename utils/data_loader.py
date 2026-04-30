"""
Utilidades para carga y preprocesamiento de datos del dataset PlantVillage.
"""

import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Devuelve las transformaciones de imagen apropiadas para cada partición.

    Args:
        split: 'train', 'val' o 'test'

    Returns:
        Composición de transformaciones de torchvision.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def split_dataset(
    dataset_path: str,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
):
    """
    Divide un ImageFolder en índices de train / val / test.

    Args:
        dataset_path: Ruta al directorio raíz del dataset (con subdirectorios
                      por clase).
        train_ratio: Fracción de ejemplos para entrenamiento.
        val_ratio: Fracción de ejemplos para validación.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla (train_indices, val_indices, test_indices).
    """
    random.seed(seed)
    full_dataset = datasets.ImageFolder(dataset_path)
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


def get_dataloaders(
    dataset_path: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
    seed: int = config.RANDOM_SEED,
):
    """
    Construye DataLoaders de train, validación y test a partir de un
    directorio con estructura ImageFolder.

    Args:
        dataset_path: Ruta al directorio raíz del dataset.
        batch_size: Tamaño de mini-batch.
        num_workers: Procesos paralelos para carga de datos.
        seed: Semilla aleatoria.

    Returns:
        Diccionario {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        y el número de clases detectadas.
    """
    train_idx, val_idx, test_idx = split_dataset(dataset_path, seed=seed)

    # Creamos tres instancias del dataset con sus propias transformaciones
    train_dataset = datasets.ImageFolder(dataset_path,
                                         transform=get_transforms("train"))
    val_dataset = datasets.ImageFolder(dataset_path,
                                       transform=get_transforms("val"))
    test_dataset = datasets.ImageFolder(dataset_path,
                                        transform=get_transforms("test"))

    loaders = {
        "train": DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            Subset(val_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            Subset(test_dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    num_classes = len(train_dataset.classes)
    return loaders, num_classes
