"""
Script de evaluación y comparación de los dos modelos entrenados.

Uso básico:
    python evaluate.py --data_dir data/PlantVillage --results_dir results/

Genera:
  • Métricas por modelo (accuracy, precision, recall, F1-macro) en test.
  • Matriz de confusión por modelo.
  • Gráfico comparativo de curvas de entrenamiento.
  • Tabla resumen en consola y en CSV.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

import config
from models.cnn_model import build_cnn
from models.resnet_model import build_resnet
from utils.data_loader import get_dataloaders
from utils.visualization import (
    plot_comparison,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_training_history,
)


# ─── Predicción sobre el conjunto de test ────────────────────────────────────

def predict(model, loader, device):
    """Realiza predicciones sobre un DataLoader y devuelve etiquetas."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluando", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─── Cálculo de métricas ─────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Devuelve un diccionario con las métricas estándar de clasificación."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred,
                                     average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred,
                               average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred,
                       average="macro", zero_division=0),
    }


# ─── Evaluación de un modelo ─────────────────────────────────────────────────

def evaluate_model(model_name, model, loaders, device, class_names,
                   results_dir, history=None):
    """
    Carga el mejor checkpoint, evalúa sobre test y genera visualizaciones.
    """
    checkpoint_path = os.path.join(results_dir, f"{model_name}_best.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device))
        print(f"  Checkpoint cargado: {checkpoint_path}")
    else:
        print(f"  AVISO: no se encontró checkpoint en {checkpoint_path}. "
              "Usando pesos actuales.")

    model = model.to(device)
    y_true, y_pred = predict(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n  Métricas [{model_name}]")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v*100:.2f}%")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    report_path = os.path.join(results_dir, f"{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Curvas de entrenamiento
    if history:
        plot_training_history(
            history, model_name,
            save_path=os.path.join(results_dir,
                                   f"{model_name}_training_curves.png"),
        )

    # Matriz de confusión
    plot_confusion_matrix(
        y_true, y_pred, class_names, model_name,
        save_path=os.path.join(results_dir,
                               f"{model_name}_confusion_matrix.png"),
    )

    return metrics


# ─── Punto de entrada ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación y comparación de modelos de detección de "
                    "enfermedades en plantas"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Ruta al dataset (estructura ImageFolder)")
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                        help="Directorio con checkpoints e historial")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    loaders, num_classes = get_dataloaders(args.data_dir, args.batch)
    from torchvision import datasets
    tmp_ds = datasets.ImageFolder(args.data_dir)
    class_names = tmp_ds.classes
    print(f"Clases: {num_classes}")

    all_metrics = {}
    all_histories = {}

    model_configs = [
        (config.MODEL_CNN, build_cnn(num_classes)),
        (config.MODEL_RESNET, build_resnet(num_classes, freeze_backbone=False)),
    ]

    for model_name, model in model_configs:
        print(f"\n{'='*60}")
        print(f"  Evaluando: {model_name}")
        print(f"{'='*60}")

        history_path = os.path.join(args.results_dir,
                                    f"{model_name}_history.json")
        history = None
        if os.path.exists(history_path):
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f)
            all_histories[model_name] = history

        metrics = evaluate_model(
            model_name, model, loaders, device, class_names,
            args.results_dir, history
        )
        all_metrics[model_name] = metrics

    # ── Comparación conjunta ─────────────────────────────────────────────
    if len(all_histories) >= 2:
        plot_comparison(
            all_histories,
            save_path=os.path.join(args.results_dir, "comparison_curves.png"),
        )

    plot_metrics_bar(
        all_metrics,
        save_path=os.path.join(args.results_dir, "comparison_metrics.png"),
    )

    # ── Tabla resumen ────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics).T * 100
    df.columns = ["Exactitud (%)", "Precisión (%)", "Recall (%)", "F1-Score (%)"]
    df = df.round(2)
    print("\n" + "="*60)
    print("  Comparación final")
    print("="*60)
    print(df.to_string())

    summary_path = os.path.join(args.results_dir, "comparison_summary.csv")
    df.to_csv(summary_path, encoding="utf-8")
    print(f"\n  Resumen guardado en: {summary_path}")

    # ── Conclusión automática ────────────────────────────────────────────
    best_model = max(all_metrics, key=lambda m: all_metrics[m]["f1"])
    print(f"\n  Mejor modelo según F1-macro: {best_model}")


if __name__ == "__main__":
    main()
