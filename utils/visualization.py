"""
Funciones de visualización para análisis y comparación de modelos.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(
    history: dict,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """
    Dibuja las curvas de pérdida y exactitud de entrenamiento y validación.

    Args:
        history: Diccionario con claves 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' (listas de valores por época).
        model_name: Nombre del modelo (para el título).
        save_path: Ruta donde guardar la figura (None = no guardar).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Pérdida
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Validación")
    axes[0].set_title(f"{model_name} — Pérdida")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida (Cross-Entropy)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Exactitud
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Entrenamiento")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Validación")
    axes[1].set_title(f"{model_name} — Exactitud")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Exactitud (%)")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_comparison(
    histories: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """
    Compara las curvas de exactitud de validación de varios modelos.

    Args:
        histories: {nombre_modelo: history_dict, ...}
        save_path: Ruta donde guardar la figura.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for i, (name, history) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        epochs = range(1, len(history["val_acc"]) + 1)
        axes[0].plot(epochs, history["val_loss"], "-o", color=c, label=name)
        axes[1].plot(epochs, history["val_acc"], "-o", color=c, label=name)

    axes[0].set_title("Comparación — Pérdida de Validación")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Comparación — Exactitud de Validación")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Exactitud (%)")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list[str],
    model_name: str,
    save_path: str | None = None,
) -> None:
    """
    Dibuja la matriz de confusión normalizada.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        class_names: Nombres de las clases.
        model_name: Nombre del modelo.
        save_path: Ruta donde guardar la figura.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    n = len(class_names)
    fig_size = max(12, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=(n <= 15),
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{model_name} — Matriz de Confusión (normalizada)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 10))
    plt.yticks(rotation=0, fontsize=max(6, 10 - n // 10))
    plt.tight_layout()
    if save_path:
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_metrics_bar(
    metrics: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """
    Gráfico de barras que compara las métricas finales de los modelos.

    Args:
        metrics: {nombre_modelo: {'accuracy': float, 'precision': float,
                                  'recall': float, 'f1': float}, ...}
        save_path: Ruta donde guardar la figura.
    """
    model_names = list(metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Exactitud", "Precisión", "Recall", "F1-Score"]

    x = np.arange(len(metric_keys))
    width = 0.35
    colors = ["steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        values = [metrics[name].get(k, 0) * 100 for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Valor (%)")
    ax.set_title("Comparación de Métricas Finales en Test")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
