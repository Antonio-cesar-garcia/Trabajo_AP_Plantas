"""
Modelo 1 — CNN personalizada para detección de enfermedades en plantas.

Arquitectura:
  Conv Block 1: Conv(3→32) → BN → ReLU → MaxPool
  Conv Block 2: Conv(32→64) → BN → ReLU → MaxPool
  Conv Block 3: Conv(64→128) → BN → ReLU → MaxPool
  Conv Block 4: Conv(128→256) → BN → ReLU → MaxPool
  Head:         AdaptiveAvgPool → Dropout(0.5) → FC(256→num_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv → BN → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    CNN personalizada de cuatro bloques convolucionales.

    Args:
        num_classes: Número de clases de salida.
        dropout_rate: Tasa de dropout en el clasificador.
    """

    def __init__(self, num_classes: int = 38,
                 dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 224 → 112
            ConvBlock(32, 64),   # 112 →  56
            ConvBlock(64, 128),  #  56 →  28
            ConvBlock(128, 256), #  28 →  14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_cnn(num_classes: int = 38) -> CustomCNN:
    """Instancia y devuelve el modelo CNN personalizado."""
    return CustomCNN(num_classes=num_classes)
