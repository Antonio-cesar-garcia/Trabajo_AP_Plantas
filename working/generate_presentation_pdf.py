from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


ROOT = Path(r"c:\Users\antca\Desktop\MasterIA\trabajo plantas")
WORKING = ROOT / "working" / "checkpoints"
OUT_PDF = ROOT / "working" / "Plantas_enfermedades2_presentacion.pdf"


PRE_PATH = WORKING / "model_comparison.csv"
POST_PATH = WORKING / "model_comparison_optimized.csv"
PRE_POST_PATH = WORKING / "pre_post_hyperparam_comparison.csv"


plt.style.use("seaborn-v0_8-whitegrid")


def _load_data() -> pd.DataFrame:
    if PRE_POST_PATH.exists():
        df = pd.read_csv(PRE_POST_PATH)
    else:
        pre = pd.read_csv(PRE_PATH).rename(
            columns={
                "Test Loss": "Test Loss (Pre)",
                "Test Acc": "Test Acc (Pre)",
                "Test F1": "Test F1 (Pre)",
                "Total Time (s)": "Total Time (s) (Pre)",
            }
        )
        post = pd.read_csv(POST_PATH).rename(
            columns={
                "Test Loss": "Test Loss (Post)",
                "Test Acc": "Test Acc (Post)",
                "Test F1": "Test F1 (Post)",
                "Total Time (s)": "Total Time (s) (Post)",
            }
        )
        df = pre.merge(post, on="Modelo", how="inner")

    df = df[df["Modelo"].isin(["resnet50", "efficientnet_b0"])].copy()
    df["Modelo"] = df["Modelo"].replace({"resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"})

    if "Test Loss" not in df.columns and "Test Loss (Post)" in df.columns:
        df["Test Loss"] = df["Test Loss (Post)"]
    if "Test Acc" not in df.columns and "Test Acc (Post)" in df.columns:
        df["Test Acc"] = df["Test Acc (Post)"]
    if "Test F1" not in df.columns and "Test F1 (Post)" in df.columns:
        df["Test F1"] = df["Test F1 (Post)"]
    if "Total Time (s)" not in df.columns and "Total Time (s) (Post)" in df.columns:
        df["Total Time (s)"] = df["Total Time (s) (Post)"]

    return df.reset_index(drop=True)


def _add_header(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.05, 0.95, title, fontsize=24, fontweight="bold", ha="left", va="top", color="#0f172a")
    if subtitle:
        fig.text(0.05, 0.905, subtitle, fontsize=11.5, ha="left", va="top", color="#334155")
    fig.text(0.95, 0.95, "Plantas_enfermedades2.ipynb", fontsize=10, ha="right", va="top", color="#64748b")


def _add_footer(fig, text: str) -> None:
    fig.text(0.95, 0.03, text, fontsize=9, ha="right", va="bottom", color="#64748b")


def _page_cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.patches.extend([
        Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="#f8fafc", zorder=-10),
        Rectangle((0, 0.84), 1, 0.16, transform=fig.transFigure, color="#0f172a", zorder=-9),
        Rectangle((0, 0), 1, 0.03, transform=fig.transFigure, color="#0f172a", zorder=-9),
    ])

    fig.text(0.06, 0.92, "Presentación técnica del notebook", fontsize=18, color="#cbd5e1", weight="bold")
    fig.text(0.06, 0.77, "Clasificación de enfermedades en plantas", fontsize=30, weight="bold", color="#0f172a")
    fig.text(
        0.06,
        0.71,
        "Resumen ejecutivo de resultados, comparativa de modelos y lectura crítica a nivel doctoral",
        fontsize=14,
        color="#334155",
    )

    fig.text(0.06, 0.58, "Puntos clave", fontsize=18, weight="bold", color="#0f172a")
    bullets = [
        "Validación realizada sobre el conjunto de validación en cada época y test final sobre datos nunca vistos.",
        "Comparación entre ResNet50 y EfficientNet-B0 con métricas de Loss, Accuracy y F1-macro.",
        "Análisis pre/post ajuste de hiperparámetros para medir calidad y eficiencia computacional.",
        "Conclusión: el mejor modelo depende del criterio de decisión entre exactitud máxima, estabilidad y coste de entrenamiento.",
    ]
    y = 0.53
    for bullet in bullets:
        fig.text(0.08, y, f"• {bullet}", fontsize=13.2, color="#1e293b")
        y -= 0.07

    fig.text(0.06, 0.16, "Dataset: PlantVillage | 38 clases | pipeline con split estratificado 70/15/15", fontsize=12.5, color="#475569")
    fig.text(0.06, 0.12, "Autoría: notebook generado en el entorno local", fontsize=11.5, color="#64748b")
    _add_footer(fig, "1")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_methodology(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Metodología y validación", "La validación se hace por época sobre validation; el test queda reservado para la evaluación final.")

    # Tarjetas con flujo explícito para evitar solapamientos visuales
    card_specs = [
        (0.06, 0.49, 0.26, 0.28, "Train", "Ajuste de pesos con backpropagation.", "#ecfccb", "#3f6212"),
        (0.37, 0.49, 0.26, 0.28, "Validation", "Selección de checkpoints y control de sobreajuste.", "#dbeafe", "#1d4ed8"),
        (0.68, 0.49, 0.26, 0.28, "Test", "Estimación final sobre datos nunca usados para decisión.", "#fae8ff", "#a21caf"),
    ]

    for x0, y0, w, h, title, body, bg, fg in card_specs:
        ax = fig.add_axes([x0, y0, w, h])
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=bg, edgecolor=fg, linewidth=1.8))
        ax.text(0.06, 0.83, title, fontsize=16, weight="bold", color=fg, va="top")
        ax.text(0.06, 0.58, body, fontsize=12.5, color="#1e293b", va="top")

    fig.text(0.32, 0.61, "→", fontsize=28, color="#64748b", weight="bold")
    fig.text(0.63, 0.61, "→", fontsize=28, color="#64748b", weight="bold")

    left = fig.add_axes([0.06, 0.16, 0.56, 0.20])
    left.axis("off")
    left_text = (
        "Diseño experimental: split estratificado 70/15/15, data augmentation solo en train, normalización ImageNet y comparación en la misma partición."
    )
    left.text(0, 0.8, left_text, fontsize=13, color="#1e293b", va="top")

    right = fig.add_axes([0.66, 0.14, 0.28, 0.24])
    right.axis("off")
    right.add_patch(Rectangle((0, 0), 1, 1, transform=right.transAxes, facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5))
    right.text(0.06, 0.82, "Lectura crítica", fontsize=14.5, weight="bold", color="#1d4ed8", va="top")
    right.text(
        0.06,
        0.60,
        "El conjunto de validación guía la selección\n"
        "del modelo; test se usa solo una vez para\n"
        "medir generalización real.",
        fontsize=11.8,
        color="#1e3a8a",
        va="top",
    )
    right.text(0.06, 0.20, "Evita leakage y hace la comparación estadísticamente interpretable.", fontsize=11.5, color="#1e3a8a", va="top")

    _add_footer(fig, "2")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_training_recipe(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Arquitecturas y receta de entrenamiento", "Ambas redes reciben la misma tarea, pero el protocolo de optimización está diseñado para respetar su distinta capacidad.")

    ax = fig.add_axes([0.06, 0.18, 0.88, 0.58])
    ax.axis("off")

    blocks = [
        (0.00, 0.40, 0.30, 0.42, "Backbone", "ResNet50 / EfficientNet-B0 preentrenadas en ImageNet.", "#f8fafc", "#0f172a"),
        (0.34, 0.40, 0.30, 0.42, "Fase 1", "Se congela el extractor y se entrena solo la cabeza clasificadora.", "#ecfeff", "#155e75"),
        (0.68, 0.40, 0.30, 0.42, "Fase 2", "Se desbloquea toda la red y se baja el learning rate para afinar.", "#fff7ed", "#9a3412"),
        (0.00, 0.00, 0.30, 0.30, "Loss", "CrossEntropy con ponderación por clase y label smoothing.", "#fef2f2", "#b91c1c"),
        (0.34, 0.00, 0.30, 0.30, "Optimizer", "AdamW con weight decay para robustez y mejor regularización.", "#f5f3ff", "#6d28d9"),
        (0.68, 0.00, 0.30, 0.30, "Scheduler", "ReduceLROnPlateau ajusta el LR cuando la validación se estanca.", "#f0fdf4", "#166534"),
    ]

    for x0, y0, w, h, title, body, bg, fg in blocks:
        block = fig.add_axes([0.06 + x0 * 0.88, 0.18 + y0 * 0.58, w * 0.88, h * 0.58])
        block.axis("off")
        block.add_patch(Rectangle((0, 0), 1, 1, transform=block.transAxes, facecolor=bg, edgecolor=fg, linewidth=1.6))
        block.text(0.06, 0.82, title, fontsize=15, weight="bold", color=fg, va="top")
        block.text(0.06, 0.56, body, fontsize=11.8, color="#1e293b", va="top")

    fig.text(0.06, 0.08, "Ritmo oral sugerido: 45-60 segundos en esta diapositiva.", fontsize=11.5, color="#475569")
    _add_footer(fig, "3")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_metrics_table(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Resultados principales en test", "Métricas finales del mejor checkpoint guardado para cada arquitectura.")

    table_ax = fig.add_axes([0.05, 0.16, 0.9, 0.64])
    table_ax.axis("off")

    cols = ["Modelo", "Test Loss", "Test Acc", "Test F1", "Total Time (s)"]
    display_df = df[cols].copy()
    display_df["Test Acc"] = (display_df["Test Acc"] * 100).round(2)
    display_df["Test F1"] = (display_df["Test F1"] * 100).round(2)
    display_df["Test Loss"] = display_df["Test Loss"].round(4)
    display_df["Total Time (s)"] = display_df["Total Time (s)"].round(1)
    display_df.columns = ["Modelo", "Loss", "Accuracy (%)", "F1-macro (%)", "Tiempo total (s)"]

    tbl = table_ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.16, 1.0, 0.64],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")

    best_acc = df.loc[df["Test Acc"].idxmax()]
    best_f1 = df.loc[df["Test F1"].idxmax()]
    best_loss = df.loc[df["Test Loss"].idxmin()]

    fig.text(0.06, 0.08, f"Mejor accuracy: {best_acc['Modelo']} ({best_acc['Test Acc']*100:.2f}%)", fontsize=12.5, color="#0f172a")
    fig.text(0.06, 0.05, f"Mejor F1: {best_f1['Modelo']} ({best_f1['Test F1']*100:.2f}%)", fontsize=12.5, color="#0f172a")
    fig.text(0.56, 0.08, f"Menor loss: {best_loss['Modelo']} ({best_loss['Test Loss']:.4f})", fontsize=12.5, color="#0f172a")
    fig.text(0.56, 0.05, "La diferencia entre modelos es pequeña en accuracy, pero no en eficiencia ni estabilidad relativa.", fontsize=11.5, color="#475569")
    _add_footer(fig, "4")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_pre_post_bars(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.333, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.28)
    _add_header(fig, "Comparación pre/post ajuste de hiperparámetros", "Barrido comparativo entre el estado previo y el optimizado para ResNet50 y EfficientNet-B0.")

    metrics = [
        ("Test Loss", ["Test Loss (Pre)", "Test Loss (Post)"], "Loss", False),
        ("Test Accuracy", ["Test Acc (Pre)", "Test Acc (Post)"], "Accuracy", True),
        ("Test F1-macro", ["Test F1 (Pre)", "Test F1 (Post)"], "F1-macro", True),
    ]

    colors = ["#94a3b8", "#0f766e"]
    for ax, (title, cols, ylabel, pct) in zip(axes, metrics):
        plot_df = df[["Modelo"] + cols].melt(id_vars="Modelo", value_vars=cols, var_name="Etapa", value_name="Valor")
        plot_df["Etapa"] = plot_df["Etapa"].map({cols[0]: "Pre", cols[1]: "Post"})
        for i, etapa in enumerate(["Pre", "Post"]):
            subset = plot_df[plot_df["Etapa"] == etapa]
            x = np.arange(len(subset["Modelo"])) + (i - 0.5) * 0.28
            ax.bar(x, subset["Valor"], width=0.28, color=colors[i], label=etapa if title == "Test Loss" else None)
        ax.set_title(title, fontsize=13.5, weight="bold")
        ax.set_xticks(np.arange(len(df["Modelo"])) )
        ax.set_xticklabels(df["Modelo"], rotation=0)
        if pct:
            ax.set_ylabel("Valor")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.1f}%")
        else:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(["Pre", "Post"], loc="upper right", frameon=True)
    _add_footer(fig, "5")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_delta_and_efficiency(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Cambio de rendimiento y eficiencia", "El cambio pre/post se analiza junto al coste computacional para evitar conclusiones unidimensionales.")

    left = fig.add_axes([0.06, 0.16, 0.42, 0.62])
    right = fig.add_axes([0.56, 0.16, 0.38, 0.62])

    deltas = pd.DataFrame({
        "Modelo": df["Modelo"],
        "Delta Loss": df["Test Loss (Post)"] - df["Test Loss (Pre)"],
        "Delta Acc (%)": (df["Test Acc (Post)"] - df["Test Acc (Pre)"]) * 100,
        "Delta F1 (%)": (df["Test F1 (Post)"] - df["Test F1 (Pre)"]) * 100,
        "Time Saved (s)": df["Total Time (s) (Pre)"] - df["Total Time (s) (Post)"],
        "Speedup (x)": df["Total Time (s) (Pre)"] / df["Total Time (s) (Post)"],
    })

    delta_cols = ["Delta Loss", "Delta Acc (%)", "Delta F1 (%)", "Time Saved (s)"]
    palette = ["#ef4444", "#14b8a6", "#0ea5e9", "#f59e0b"]
    left.barh(deltas["Modelo"], deltas["Delta Acc (%)"], color=palette[1], alpha=0.95, label="Accuracy")
    left.barh(deltas["Modelo"], deltas["Delta F1 (%)"], color=palette[2], alpha=0.7, label="F1")
    left.set_title("Ganancia/retroceso relativo en métricas", fontsize=13.5, weight="bold")
    left.axvline(0, color="#0f172a", linewidth=1)
    left.set_xlabel("Cambio porcentual (p.p.)")
    left.legend(frameon=True, loc="lower right")
    left.grid(axis="x", alpha=0.25)

    time_df = df[["Modelo", "Total Time (s) (Pre)", "Total Time (s) (Post)"]].copy()
    x = np.arange(len(time_df["Modelo"]))
    width = 0.34
    right.bar(x - width / 2, time_df["Total Time (s) (Pre)"], width=width, color="#94a3b8", label="Pre")
    right.bar(x + width / 2, time_df["Total Time (s) (Post)"], width=width, color="#0f766e", label="Post")
    right.set_xticks(x)
    right.set_xticklabels(time_df["Modelo"])
    right.set_ylabel("Tiempo total (s)")
    right.set_title("Coste computacional", fontsize=13.5, weight="bold")
    right.legend(frameon=True)
    right.grid(axis="y", alpha=0.25)

    fig.text(0.06, 0.08, f"Speedup ResNet50: {deltas.loc[deltas['Modelo']=='ResNet50', 'Speedup (x)'].iloc[0]:.2f}x", fontsize=11.8, color="#0f172a")
    fig.text(0.32, 0.08, f"Speedup EfficientNet-B0: {deltas.loc[deltas['Modelo']=='EfficientNet-B0', 'Speedup (x)'].iloc[0]:.2f}x", fontsize=11.8, color="#0f172a")
    fig.text(0.60, 0.08, "La mejora no es uniforme: un ajuste puede beneficiar de forma distinta a cada arquitectura.", fontsize=11.5, color="#475569")
    _add_footer(fig, "6")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_frontier_and_conclusions(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Lectura doctoral de los resultados", "No basta con maximizar accuracy: importa la relación entre calidad, estabilidad y coste.")

    ax = fig.add_axes([0.06, 0.2, 0.38, 0.56])
    x = df["Total Time (s) (Post)"]
    y = df["Test Acc (Post)"] * 100
    ax.scatter(x, y, s=180, color=["#0f766e", "#f97316"], alpha=0.9)
    for _, row in df.iterrows():
        ax.annotate(row["Modelo"], (row["Total Time (s) (Post)"], row["Test Acc (Post)"] * 100), xytext=(6, 6), textcoords="offset points", fontsize=11)
    ax.set_xlabel("Tiempo post (s)")
    ax.set_ylabel("Accuracy post (%)")
    ax.set_title("Frente calidad-coste", fontsize=13.5, weight="bold")
    ax.grid(alpha=0.25)

    txt = fig.add_axes([0.50, 0.18, 0.44, 0.62])
    txt.axis("off")

    winner_acc = df.loc[df["Test Acc"].idxmax(), "Modelo"]
    winner_f1 = df.loc[df["Test F1"].idxmax(), "Modelo"]
    winner_time = df.loc[df["Total Time (s)"].idxmin(), "Modelo"]
    text = (
        f"Conclusiones clave\n\n"
        f"• En el estado optimizado, {winner_acc} concentra el mejor accuracy final, pero la ventaja frente a la otra arquitectura es pequeña.\n"
        f"• {winner_f1} mantiene el mejor F1, que es una métrica más robusta cuando hay desbalance entre clases.\n"
        f"• {winner_time} es más eficiente en tiempo total, por lo que domina si el objetivo es coste computacional.\n\n"
        f"Lectura metodológica\n\n"
        f"• La validación por época en el conjunto de validación es la base para seleccionar checkpoints sin contaminar el test.\n"
        f"• El rendimiento debe interpretarse con la incertidumbre implícita del proceso de fine-tuning y no como una verdad absoluta del modelo.\n"
        f"• La ligera asimetría entre métricas sugiere que la elección final depende del criterio operativo: sensibilidad global, estabilidad por clase o eficiencia de entrenamiento.\n"
    )
    txt.text(0, 1, text, fontsize=12.5, va="top", color="#1e293b")

    fig.text(0.06, 0.08, "Mensaje final: el mejor modelo no es solo el más preciso, sino el que optimiza el compromiso entre generalización, estabilidad y coste.", fontsize=12.2, color="#0f172a", weight="bold")
    _add_footer(fig, "7")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_closing(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    _add_header(fig, "Cierre y mensaje final", "Síntesis para cerrar en menos de un minuto.")

    ax = fig.add_axes([0.08, 0.20, 0.84, 0.54])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#0f172a", edgecolor="#0f172a", linewidth=1.2))
    ax.text(0.05, 0.80, "Qué defendería en la exposición", fontsize=17, weight="bold", color="white", va="top")
    closing = (
        "1. La validación se hace durante el entrenamiento sobre validation; test queda fuera de la toma de decisiones.\n"
        "2. EfficientNet-B0 tiende a dominar en calidad pura, mientras que ResNet50 ofrece una referencia más estable en algunos escenarios.\n"
        "3. La optimización de hiperparámetros cambia el equilibrio entre rendimiento y coste, no solo la accuracy.\n"
        "4. La métrica correcta depende del objetivo: F1-macro si preocupa el desbalance, accuracy si se busca simplicidad, y tiempo si importa la eficiencia."
    )
    ax.text(0.05, 0.63, closing, fontsize=13.0, color="#e2e8f0", va="top", linespacing=1.55)

    fig.text(0.08, 0.10, "Duración objetivo total: 8 minutos aproximadamente (7 diapositivas de 45-60 s y un cierre breve).", fontsize=12, color="#0f172a")
    _add_footer(fig, "8")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf() -> Path:
    df = _load_data()
    with PdfPages(OUT_PDF) as pdf:
        _page_cover(pdf)
        _page_methodology(pdf)
        _page_training_recipe(pdf)
        _page_metrics_table(pdf, df)
        _page_pre_post_bars(pdf, df)
        _page_delta_and_efficiency(pdf, df)
        _page_frontier_and_conclusions(pdf, df)
        _page_closing(pdf, df)
    return OUT_PDF


if __name__ == "__main__":
    out = build_pdf()
    print(out)