"""
Modelo 2 — ResNet-50 con aprendizaje por transferencia para detección
de enfermedades en plantas.

Estrategia:
  • Se carga ResNet-50 pre-entrenado en ImageNet.
  • En la primera fase se congela el extractor de características y solo
    se entrena la capa final (fine-tuning superficial).
  • En la segunda fase (opcional) se descongelan los últimos bloques
    para un ajuste fino más profundo.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetTransfer(nn.Module):
    """
    ResNet-50 adaptado para clasificación de enfermedades en plantas.

    Args:
        num_classes: Número de clases de salida.
        freeze_backbone: Si True, congela todos los parámetros excepto
                         la capa final (útil para el warm-up inicial).
        dropout_rate: Dropout aplicado antes de la capa final.
    """

    def __init__(
        self,
        num_classes: int = 38,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Reemplazar el clasificador original
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

        if freeze_backbone:
            self._freeze_backbone()

    # ── Control de parámetros entrenables ──────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Congela todos los parámetros excepto la capa final."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_layer4(self) -> None:
        """Descongela el último bloque residual (layer4) y la capa FC."""
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Descongela todos los parámetros del modelo."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list[dict]:
        """
        Devuelve grupos de parámetros entrenables para el optimizador,
        con distinta tasa de aprendizaje según profundidad.
        """
        fc_params = list(self.model.fc.parameters())
        fc_ids = set(id(p) for p in fc_params)

        backbone_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in fc_ids
        ]
        return [
            {"params": backbone_params, "lr": 1e-4},
            {"params": fc_params, "lr": 1e-3},
        ]

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_resnet(num_classes: int = 38,
                 freeze_backbone: bool = True) -> ResNetTransfer:
    """Instancia y devuelve el modelo ResNet-50 con transfer learning."""
    return ResNetTransfer(num_classes=num_classes,
                          freeze_backbone=freeze_backbone)
