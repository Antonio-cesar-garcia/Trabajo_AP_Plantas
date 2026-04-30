"""
Script de entrenamiento para los dos modelos de detección de enfermedades
en plantas (CNN personalizada y ResNet-50 con transfer learning).

Uso básico:
    python train.py --data_dir data/PlantVillage --model cnn
    python train.py --data_dir data/PlantVillage --model resnet
    python train.py --data_dir data/PlantVillage --model all

Argumentos:
    --data_dir  : Ruta al directorio raíz del dataset (estructura ImageFolder).
    --model     : Modelo a entrenar: 'cnn', 'resnet' o 'all'.
    --epochs    : Número máximo de épocas (por defecto: config.NUM_EPOCHS).
    --batch     : Tamaño de batch      (por defecto: config.BATCH_SIZE).
    --lr        : Tasa de aprendizaje  (por defecto: config.LEARNING_RATE).
    --output    : Directorio de salida (por defecto: results/).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from models.cnn_model import build_cnn
from models.resnet_model import build_resnet
from utils.data_loader import get_dataloaders


# ─── Entrenamiento de una época ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ─── Evaluación de una época ─────────────────────────────────────────────────

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ─── Bucle de entrenamiento completo ─────────────────────────────────────────

def train_model(
    model,
    model_name: str,
    loaders: dict,
    num_epochs: int,
    learning_rate: float,
    output_dir: str,
    device: torch.device,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    unfreeze_epoch: int = 5,
):
    """
    Entrena un modelo y guarda el mejor checkpoint.

    Para ResNet se usa una estrategia de dos fases:
      • Épocas 1..unfreeze_epoch : solo se entrena la capa FC.
      • Épocas >unfreeze_epoch   : se descongela layer4 y se reduce el lr.

    Returns:
        Diccionario con el historial de entrenamiento.
    """
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # Fase 1: cabeza del clasificador (o todos los params si es CNN)
    if hasattr(model, "get_trainable_params"):
        param_groups = model.get_trainable_params()
    else:
        param_groups = model.parameters()

    optimizer = optim.Adam(param_groups, lr=learning_rate,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pth")

    print(f"\n{'='*60}")
    print(f"  Entrenando: {model_name}")
    print(f"  Épocas: {num_epochs}  |  LR: {learning_rate}  |  Device: {device}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── Segunda fase: descongelar para fine-tuning profundo ──────────
        if hasattr(model, "unfreeze_layer4") and epoch == unfreeze_epoch + 1:
            print(f"\n  [Época {epoch}] Descongelando layer4 para fine-tuning…")
            model.unfreeze_layer4()
            # Reiniciar optimizador usando grupos de parámetros con LR diferenciado
            param_groups = model.get_trainable_params()
            # Escalar las tasas de aprendizaje para el fine-tuning profundo
            for group in param_groups:
                group["lr"] = group.get("lr", learning_rate) * 0.1
            optimizer = optim.Adam(param_groups,
                                   weight_decay=config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                          patience=3, verbose=True)

        train_loss, train_acc = train_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(
            model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(
            f"  Época {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}%"
        )

        # ── Early stopping + checkpoint ──────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= patience:
            print(f"\n  Early stopping en época {epoch} "
                  f"(mejor época: {best_epoch})")
            break

    elapsed = time.time() - start_time
    print(f"\n  Tiempo total: {elapsed/60:.1f} min")
    print(f"  Mejor época: {best_epoch}  |  Mejor val loss: {best_val_loss:.4f}")

    # Guardar historial
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


# ─── Punto de entrada ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos para detección de enfermedades "
                    "en plantas"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Ruta al dataset (estructura ImageFolder)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["cnn", "resnet", "all"],
                        help="Modelo a entrenar")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--output", type=str, default=config.RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Dispositivo: {device}")

    # Construir dataloaders
    loaders, num_classes = get_dataloaders(args.data_dir, args.batch)
    print(f"Clases detectadas: {num_classes}")

    models_to_train = []
    if args.model in ("cnn", "all"):
        models_to_train.append((config.MODEL_CNN, build_cnn(num_classes)))
    if args.model in ("resnet", "all"):
        models_to_train.append(
            (config.MODEL_RESNET, build_resnet(num_classes, freeze_backbone=True))
        )

    for model_name, model in models_to_train:
        model = model.to(device)
        train_model(
            model=model,
            model_name=model_name,
            loaders=loaders,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output,
            device=device,
        )

    print("\n✓ Entrenamiento completado. Resultados en:", args.output)


if __name__ == "__main__":
    main()
