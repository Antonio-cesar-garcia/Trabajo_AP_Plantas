from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import fill
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"c:\Users\antca\Desktop\MasterIA\trabajo plantas")
WORKING = ROOT / "working" / "checkpoints"
BUILD_DIR = ROOT / "working" / "pptx_build"
SLIDE_DIR = BUILD_DIR / "slides"
OUT_PPTX = ROOT / "working" / "Plantas_enfermedades2_presentacion_fixed2.pptx"

PRE_PATH = WORKING / "model_comparison.csv"
POST_PATH = WORKING / "model_comparison_optimized.csv"
PRE_POST_PATH = WORKING / "pre_post_hyperparam_comparison.csv"

SLIDE_W_IN = 13.333333
SLIDE_H_IN = 7.5
DPI = 160
SLIDE_W_PX = int(SLIDE_W_IN * DPI)
SLIDE_H_PX = int(SLIDE_H_IN * DPI)
SLIDE_W_EMU = 12192000
SLIDE_H_EMU = 6858000


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


def _base_figure(facecolor: str = "white"):
    fig = plt.figure(figsize=(SLIDE_W_IN, SLIDE_H_IN), dpi=DPI, facecolor=facecolor)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _title(fig, title: str, subtitle: str | None = None, dark: bool = False) -> None:
    color = "white" if dark else "#0f172a"
    sub_color = "#cbd5e1" if dark else "#334155"
    fig.text(0.055, 0.95, title, fontsize=24, fontweight="bold", ha="left", va="top", color=color)
    if subtitle:
        fig.text(0.055, 0.905, subtitle, fontsize=11.5, ha="left", va="top", color=sub_color)
    fig.text(0.965, 0.95, "Plantas_enfermedades2.ipynb", fontsize=10, ha="right", va="top", color=sub_color)


def _footer(fig, n: int, dark: bool = False) -> None:
    fig.text(0.965, 0.03, str(n), fontsize=9, ha="right", va="bottom", color=("#cbd5e1" if dark else "#64748b"))


def _text_box(fig, x, y, w, h, facecolor, edgecolor, title, body, title_color=None, body_color="#1e293b"):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6))
    ax.text(0.06, 0.84, title, fontsize=15, weight="bold", color=title_color or edgecolor, va="top")
    wrapped_body = fill(body, width=max(24, int(w * 122)))
    ax.text(0.06, 0.58, wrapped_body, fontsize=10.8, color=body_color, va="top", linespacing=1.22)
    return ax


def _slide_cover(out: Path) -> None:
    fig = _base_figure()
    fig.patches.extend([
        plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="#f8fafc", zorder=-10),
        plt.Rectangle((0, 0.84), 1, 0.16, transform=fig.transFigure, color="#0f172a", zorder=-9),
        plt.Rectangle((0, 0), 1, 0.03, transform=fig.transFigure, color="#0f172a", zorder=-9),
    ])
    fig.text(0.06, 0.92, "Presentacion tecnica del notebook", fontsize=18, color="#cbd5e1", weight="bold")
    fig.text(0.06, 0.77, "Clasificacion de enfermedades en plantas", fontsize=30, weight="bold", color="#0f172a")
    fig.text(0.06, 0.71, "Resumen ejecutivo de resultados, comparativa de modelos y lectura critica a nivel doctoral", fontsize=14, color="#334155")
    fig.text(0.06, 0.58, "Puntos clave", fontsize=18, weight="bold", color="#0f172a")
    bullets = [
        "Validacion por epoca sobre validation y test final solo para evaluacion.",
        "Comparacion entre ResNet50 y EfficientNet-B0 en Loss, Accuracy y F1-macro.",
        "Analisis pre/post ajuste de hiperparametros para balancear calidad y coste.",
        "La eleccion final depende del criterio operativo: precision, estabilidad o eficiencia.",
    ]
    y = 0.53
    for bullet in bullets:
        fig.text(0.08, y, f"• {bullet}", fontsize=13.2, color="#1e293b")
        y -= 0.07
    fig.text(0.06, 0.16, "Dataset: PlantVillage | 38 clases | split estratificado 70/15/15", fontsize=12.5, color="#475569")
    fig.text(0.06, 0.12, "Presentacion de 8 minutos aprox.", fontsize=11.5, color="#64748b")
    _footer(fig, 1)
    _save_fig(fig, out)


def _slide_methodology(out: Path) -> None:
    fig = _base_figure()
    _title(fig, "Metodologia y validacion", "La validacion se hace por epoca sobre validation; el test queda reservado para la evaluacion final.")

    specs = [
        (0.06, 0.50, 0.26, 0.27, "Train", "Ajuste de pesos con backpropagation.", "#ecfccb", "#3f6212"),
        (0.37, 0.50, 0.26, 0.27, "Validation", "Seleccion de checkpoints y control de sobreajuste.", "#dbeafe", "#1d4ed8"),
        (0.68, 0.50, 0.26, 0.27, "Test", "Estimacion final sobre datos nunca usados para decidir.", "#fae8ff", "#a21caf"),
    ]
    for x, y, w, h, title, body, bg, fg in specs:
        _text_box(fig, x, y, w, h, bg, fg, title, body)

    fig.text(0.32, 0.61, "→", fontsize=28, color="#64748b", weight="bold")
    fig.text(0.63, 0.61, "→", fontsize=28, color="#64748b", weight="bold")

    ax = fig.add_axes([0.06, 0.14, 0.52, 0.20])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.1))
    ax.text(0.03, 0.82, "Diseño experimental", fontsize=12.5, color="#0f172a", weight="bold", va="top")
    ax.text(
      0.03,
      0.60,
      fill(
        "Split estratificado 70/15/15, data augmentation solo en train, normalizacion ImageNet y comparacion en la misma particion.",
        width=74,
      ),
      fontsize=11.0,
      color="#1e293b",
      va="top",
      linespacing=1.35,
    )

    ax2 = fig.add_axes([0.61, 0.14, 0.33, 0.20])
    ax2.axis("off")
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5))
    ax2.text(0.06, 0.82, "Lectura critica", fontsize=14.5, weight="bold", color="#1d4ed8", va="top")
    ax2.text(
      0.06,
      0.58,
      fill("Validation guía la seleccion del modelo; test solo mide generalizacion real.", width=44),
      fontsize=11.0,
      color="#1e3a8a",
      va="top",
      linespacing=1.25,
    )
    ax2.text(
      0.06,
      0.20,
      fill("Evita leakage y mantiene la comparacion interpretable.", width=44),
      fontsize=10.9,
      color="#1e3a8a",
      va="top",
      linespacing=1.25,
    )
    _footer(fig, 2)
    _save_fig(fig, out)


def _slide_training_recipe(out: Path) -> None:
    fig = _base_figure()
    _title(fig, "Arquitecturas y receta de entrenamiento", "Ambas redes reciben la misma tarea, pero el protocolo de optimizacion respeta su distinta capacidad.")
    blocks = [
        (0.06, 0.42, 0.28, 0.26, "Backbone", "ResNet50 / EfficientNet-B0 preentrenadas en ImageNet.", "#f8fafc", "#0f172a"),
        (0.37, 0.42, 0.28, 0.26, "Fase 1", "Se congela el extractor y se entrena solo la cabeza.", "#ecfeff", "#155e75"),
        (0.68, 0.42, 0.26, 0.26, "Fase 2", "Se desbloquea toda la red y se baja el learning rate.", "#fff7ed", "#9a3412"),
        (0.06, 0.10, 0.28, 0.22, "Loss", "CrossEntropy con ponderacion por clase y label smoothing.", "#fef2f2", "#b91c1c"),
        (0.37, 0.10, 0.28, 0.22, "Optimizer", "AdamW con weight decay para mayor robustez.", "#f5f3ff", "#6d28d9"),
        (0.68, 0.10, 0.26, 0.22, "Scheduler", "ReduceLROnPlateau ajusta el LR cuando la validacion se estanca.", "#f0fdf4", "#166534"),
    ]
    for x, y, w, h, title, body, bg, fg in blocks:
        _text_box(fig, x, y, w, h, bg, fg, title, body)
    fig.text(0.06, 0.05, "Ritmo oral sugerido: 45-60 s.", fontsize=11.5, color="#475569")
    _footer(fig, 3)
    _save_fig(fig, out)


def _slide_results_table(out: Path, df: pd.DataFrame) -> None:
    fig = _base_figure()
    _title(fig, "Resultados principales en test", "Metricas finales del mejor checkpoint guardado para cada arquitectura.")
    ax = fig.add_axes([0.05, 0.16, 0.90, 0.64])
    ax.axis("off")
    display_df = df[["Modelo", "Test Loss", "Test Acc", "Test F1", "Total Time (s)"]].copy()
    display_df["Test Acc"] = (display_df["Test Acc"] * 100).round(2)
    display_df["Test F1"] = (display_df["Test F1"] * 100).round(2)
    display_df["Test Loss"] = display_df["Test Loss"].round(4)
    display_df["Total Time (s)"] = display_df["Total Time (s)"].round(1)
    display_df.columns = ["Modelo", "Loss", "Accuracy (%)", "F1-macro (%)", "Tiempo total (s)"]
    tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center", cellLoc="center", colLoc="center", bbox=[0, 0.16, 1, 0.64])
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
    best_acc = df.loc[df["Test Acc"].idxmax()]
    best_f1 = df.loc[df["Test F1"].idxmax()]
    best_loss = df.loc[df["Test Loss"].idxmin()]
    fig.text(0.06, 0.08, f"Mejor accuracy: {best_acc['Modelo']} ({best_acc['Test Acc']*100:.2f}%)", fontsize=12.5, color="#0f172a")
    fig.text(0.06, 0.05, f"Mejor F1: {best_f1['Modelo']} ({best_f1['Test F1']*100:.2f}%)", fontsize=12.5, color="#0f172a")
    fig.text(0.56, 0.08, f"Menor loss: {best_loss['Modelo']} ({best_loss['Test Loss']:.4f})", fontsize=12.5, color="#0f172a")
    fig.text(0.56, 0.05, "La diferencia en accuracy es pequena, pero no en eficiencia ni estabilidad relativa.", fontsize=11.5, color="#475569")
    _footer(fig, 4)
    _save_fig(fig, out)


def _slide_prepost(out: Path, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(SLIDE_W_IN, SLIDE_H_IN), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.28)
    _title(fig, "Comparacion pre/post de hiperparametros", "Barrido comparativo entre el estado previo y el optimizado para ambas arquitecturas.")
    metrics = [
        ("Test Loss", ["Test Loss (Pre)", "Test Loss (Post)"], False),
        ("Test Accuracy", ["Test Acc (Pre)", "Test Acc (Post)"], True),
        ("Test F1-macro", ["Test F1 (Pre)", "Test F1 (Post)"], True),
    ]
    colors = ["#94a3b8", "#0f766e"]
    for ax, (title, cols, pct) in zip(axes, metrics):
        plot_df = df[["Modelo"] + cols].melt(id_vars="Modelo", value_vars=cols, var_name="Etapa", value_name="Valor")
        plot_df["Etapa"] = plot_df["Etapa"].map({cols[0]: "Pre", cols[1]: "Post"})
        for i, etapa in enumerate(["Pre", "Post"]):
            subset = plot_df[plot_df["Etapa"] == etapa]
            x = np.arange(len(subset["Modelo"])) + (i - 0.5) * 0.28
            ax.bar(x, subset["Valor"], width=0.28, color=colors[i], label=etapa if title == "Test Loss" else None)
        ax.set_title(title, fontsize=13.5, weight="bold")
        ax.set_xticks(np.arange(len(df["Modelo"])))
        ax.set_xticklabels(df["Modelo"])  
        if pct:
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.1f}%")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(["Pre", "Post"], loc="upper right", frameon=True)
    _footer(fig, 5)
    _save_fig(fig, out)


def _slide_delta_efficiency(out: Path, df: pd.DataFrame) -> None:
    fig = _base_figure()
    _title(fig, "Cambio de rendimiento y eficiencia", "El cambio pre/post se analiza junto al coste computacional para evitar conclusiones unidimensionales.")
    left = fig.add_axes([0.06, 0.16, 0.42, 0.62])
    right = fig.add_axes([0.56, 0.16, 0.38, 0.62])
    deltas = pd.DataFrame({
        "Modelo": df["Modelo"],
        "Delta Acc (%)": (df["Test Acc (Post)"] - df["Test Acc (Pre)"]) * 100,
        "Delta F1 (%)": (df["Test F1 (Post)"] - df["Test F1 (Pre)"]) * 100,
        "Speedup (x)": df["Total Time (s) (Pre)"] / df["Total Time (s) (Post)"],
    })
    left.barh(deltas["Modelo"], deltas["Delta Acc (%)"], color="#14b8a6", alpha=0.95, label="Accuracy")
    left.barh(deltas["Modelo"], deltas["Delta F1 (%)"], color="#0ea5e9", alpha=0.70, label="F1")
    left.set_title("Ganancia/retroceso relativo en metricas", fontsize=13.5, weight="bold")
    left.axvline(0, color="#0f172a", linewidth=1)
    left.set_xlabel("Cambio porcentual (p.p.)")
    left.legend(frameon=True, loc="lower right")
    left.grid(axis="x", alpha=0.25)
    x = np.arange(len(df["Modelo"]))
    width = 0.34
    right.bar(x - width / 2, df["Total Time (s) (Pre)"], width=width, color="#94a3b8", label="Pre")
    right.bar(x + width / 2, df["Total Time (s) (Post)"], width=width, color="#0f766e", label="Post")
    right.set_xticks(x)
    right.set_xticklabels(df["Modelo"])
    right.set_ylabel("Tiempo total (s)")
    right.set_title("Coste computacional", fontsize=13.5, weight="bold")
    right.legend(frameon=True)
    right.grid(axis="y", alpha=0.25)
    fig.text(0.06, 0.08, f"Speedup ResNet50: {deltas.loc[deltas['Modelo']=='ResNet50', 'Speedup (x)'].iloc[0]:.2f}x", fontsize=11.8, color="#0f172a")
    fig.text(0.32, 0.08, f"Speedup EfficientNet-B0: {deltas.loc[deltas['Modelo']=='EfficientNet-B0', 'Speedup (x)'].iloc[0]:.2f}x", fontsize=11.8, color="#0f172a")
    fig.text(0.60, 0.08, "La mejora no es uniforme: un ajuste puede beneficiar de forma distinta a cada arquitectura.", fontsize=11.5, color="#475569")
    _footer(fig, 6)
    _save_fig(fig, out)


def _slide_frontier(out: Path, df: pd.DataFrame) -> None:
    fig = _base_figure()
    _title(fig, "Lectura doctoral de los resultados", "No basta con maximizar accuracy: importa la relacion entre calidad, estabilidad y coste.")
    ax = fig.add_axes([0.06, 0.2, 0.38, 0.56])
    ax.scatter(df["Total Time (s) (Post)"], df["Test Acc (Post)"] * 100, s=180, color=["#0f766e", "#f97316"], alpha=0.9)
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
        f"• En el estado optimizado, {winner_acc} concentra el mejor accuracy final, pero la ventaja es pequena.\n"
        f"• {winner_f1} mantiene el mejor F1, mas robusto cuando hay desbalance entre clases.\n"
        f"• {winner_time} es mas eficiente en tiempo total, por lo que domina si el objetivo es coste computacional.\n\n"
        f"Lectura metodologica\n\n"
        f"• Validation gobierna la seleccion de checkpoints; test se usa una sola vez.\n"
        f"• El rendimiento debe interpretarse junto con la estabilidad del fine-tuning, no como verdad absoluta.\n"
        f"• La eleccion final depende del criterio operativo: sensibilidad global, estabilidad por clase o eficiencia.\n"
    )
    txt.text(0, 1, text, fontsize=12.5, va="top", color="#1e293b")
    fig.text(0.06, 0.08, "Mensaje final: el mejor modelo no es solo el mas preciso, sino el que optimiza el compromiso entre generalizacion, estabilidad y coste.", fontsize=12.2, color="#0f172a", weight="bold")
    _footer(fig, 7)
    _save_fig(fig, out)


def _slide_closing(out: Path) -> None:
    fig = _base_figure()
    _title(fig, "Cierre y mensaje final", "Sintesis para cerrar en menos de un minuto.")
    ax = fig.add_axes([0.08, 0.20, 0.84, 0.54])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#0f172a", edgecolor="#0f172a", linewidth=1.2))
    ax.text(0.05, 0.80, "Que defenderia en la exposicion", fontsize=17, weight="bold", color="white", va="top")
    closing = (
        "1. La validacion se hace durante el entrenamiento sobre validation; test queda fuera de la toma de decisiones.\n"
        "2. EfficientNet-B0 tiende a dominar en calidad pura, mientras que ResNet50 ofrece una referencia estable.\n"
        "3. La optimizacion de hiperparametros cambia el equilibrio entre rendimiento y coste, no solo la accuracy.\n"
        "4. La metrica correcta depende del objetivo: F1-macro, accuracy o tiempo."
    )
    ax.text(0.05, 0.63, closing, fontsize=13.0, color="#e2e8f0", va="top", linespacing=1.55)
    fig.text(0.08, 0.10, "Duracion objetivo total: 8 minutos aproximadamente (7 diapositivas de 45-60 s y un cierre breve).", fontsize=12, color="#0f172a")
    _footer(fig, 8)
    _save_fig(fig, out)


def _theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
      <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="44546A"/></a:dk2>
      <a:lt2><a:srgbClr val="E7E6E6"/></a:lt2>
      <a:accent1><a:srgbClr val="5B9BD5"/></a:accent1>
      <a:accent2><a:srgbClr val="ED7D31"/></a:accent2>
      <a:accent3><a:srgbClr val="A5A5A5"/></a:accent3>
      <a:accent4><a:srgbClr val="FFC000"/></a:accent4>
      <a:accent5><a:srgbClr val="4472C4"/></a:accent5>
      <a:accent6><a:srgbClr val="70AD47"/></a:accent6>
      <a:hlink><a:srgbClr val="0563C1"/></a:hlink>
      <a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office">
      <a:majorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Office">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"><a:tint val="50000"/></a:schemeClr></a:gs><a:gs pos="35000"><a:schemeClr val="phClr"><a:tint val="37000"/></a:schemeClr></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"><a:tint val="15000"/></a:schemeClr></a:gs></a:gsLst><a:lin ang="16200000" scaled="1"/></a:gradFill>
        <a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"><a:shade val="51000"/></a:schemeClr></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"><a:shade val="13000"/></a:schemeClr></a:gs></a:gsLst><a:lin ang="16200000" scaled="0"/></a:gradFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
        <a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:tint val="95000"/></a:schemeClr></a:solidFill><a:solidFill><a:schemeClr val="phClr"><a:tint val="93000"/></a:schemeClr></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def _content_types_xml(slide_count: int) -> str:
    overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    image_types = '  <Default Extension="png" ContentType="image/png"/>'
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
{image_types}
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{overrides}
</Types>
"""


def _rels_root_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""


def _presentation_xml(slide_count: int) -> str:
    slide_ids = []
    for i in range(1, slide_count + 1):
        rel_id = f"rId{i + 1}"
        slide_id = 255 + i
        slide_ids.append(f'    <p:sldId id="{slide_id}" r:id="{rel_id}"/>')
    slide_ids_xml = "\n".join(slide_ids)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{slide_ids_xml}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""


def _presentation_rels_xml(slide_count: int) -> str:
    rels = [
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, slide_count + 1):
        rels.append(
            f'  <Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
""" + "\n".join(rels) + "\n</Relationships>\n"


def _slide_master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles/>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldMaster>
"""


def _slide_master_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def _slide_layout_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldLayout>
"""


def _slide_xml(image_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Slide image"/>
          <p:cNvPicPr>
            <a:picLocks noChangeAspect="1"/>
          </p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def _slide_rels_xml(image_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>
</Relationships>
"""


def _docprops_core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Plantas_enfermedades2</dc:title>
  <dc:creator>GitHub Copilot</dc:creator>
  <cp:lastModifiedBy>GitHub Copilot</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-04-30T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-04-30T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


def _docprops_app_xml(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft PowerPoint</Application>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{slide_count}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{slide_count}" baseType="lpstr">
      {''.join('<vt:lpstr>Slide</vt:lpstr>' for _ in range(slide_count))}
    </vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def _package_pptx(image_paths: list[Path], out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()

    with ZipFile(out_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(image_paths)))
        zf.writestr("_rels/.rels", _rels_root_xml())
        zf.writestr("ppt/presentation.xml", _presentation_xml(len(image_paths)))
        zf.writestr("ppt/_rels/presentation.xml.rels", _presentation_rels_xml(len(image_paths)))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", _slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _slide_master_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", _slide_layout_xml())
        zf.writestr("ppt/theme/theme1.xml", _theme_xml())
        zf.writestr("docProps/core.xml", _docprops_core_xml())
        zf.writestr("docProps/app.xml", _docprops_app_xml(len(image_paths)))

        for idx, image_path in enumerate(image_paths, start=1):
            image_name = f"image{idx}.png"
            zf.write(image_path, arcname=f"ppt/media/{image_name}")
            zf.writestr(f"ppt/slides/slide{idx}.xml", _slide_xml(image_name))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", _slide_rels_xml(image_name))


def build_pptx() -> Path:
    df = _load_data()
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)

    slide_paths: list[Path] = []
    generators = [
        _slide_cover,
        _slide_methodology,
        _slide_training_recipe,
        lambda out: _slide_results_table(out, df),
        lambda out: _slide_prepost(out, df),
        lambda out: _slide_delta_efficiency(out, df),
        lambda out: _slide_frontier(out, df),
        _slide_closing,
    ]

    for i, generator in enumerate(generators, start=1):
        slide_path = SLIDE_DIR / f"slide_{i}.png"
        generator(slide_path)
        slide_paths.append(slide_path)

    _package_pptx(slide_paths, OUT_PPTX)
    return OUT_PPTX


if __name__ == "__main__":
    out = build_pptx()
    print(out)