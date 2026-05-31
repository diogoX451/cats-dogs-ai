"""
Dashboard visual CatDogNet v2 — versão redesenhada.

Layout:
  Faixa superior : 4 KPI cards (acurácia, F1 cats, F1 dogs, alta confiança)
  Linha inferior : matriz de confusão | barras de métricas | curva ROC | histograma confiança

Uso:
    python3 plot_results.py
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, roc_curve, auc
)
from torchvision import datasets, transforms

from train import CatDogNet, IMG_SIZE

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_scratch.pth")
DATA_DIR   = Path("dataset/val")
OUTPUT     = Path("results_dashboard.png")

# ── Paleta ────────────────────────────────────────────────────────────────────
C_CATS   = "#5B8FF9"   # azul
C_DOGS   = "#F4664A"   # laranja-vermelho
C_DARK   = "#1a1a2e"
C_CARD   = "#f7f9fc"
C_GREEN  = "#30bf78"
C_AMBER  = "#faad14"

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> CatDogNet:
    model = CatDogNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def collect() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    model = load_model()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        p = torch.sigmoid(model(imgs.to(DEVICE))).squeeze(1)
        y_true.extend(labels.tolist())
        y_pred.extend((p >= 0.5).long().cpu().tolist())
        y_prob.extend(p.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ── Painéis ───────────────────────────────────────────────────────────────────

def kpi_card(ax: plt.Axes, value: str, label: str, color: str) -> None:
    ax.set_facecolor(C_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e6ef")
        spine.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.22, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="#666")


def plot_confusion(ax: plt.Axes, y_true, y_pred, classes) -> None:
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_pct, annot=False, fmt=".1f", cmap="Blues",
        linewidths=2, linecolor="white",
        cbar_kws={"format": "%.0f%%", "shrink": 0.85},
        ax=ax, vmin=0, vmax=100,
        xticklabels=classes, yticklabels=classes,
    )
    ax.set_xlabel("Predito", fontsize=12, labelpad=8)
    ax.set_ylabel("Real", fontsize=12, labelpad=8)
    ax.set_title("Matriz de Confusão", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)

    labels = [["VP", "FP"], ["FN", "VN"]]   # perspectiva cats=positivo
    thresh = 50.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > thresh else C_DARK
            ax.text(j + 0.5, i + 0.38, f"{cm_pct[i,j]:.1f}%",
                    ha="center", va="center", fontsize=16, color=color, fontweight="bold")
            ax.text(j + 0.5, i + 0.65, f"({cm[i,j]:,})",
                    ha="center", va="center", fontsize=10, color=color, alpha=0.85)
            ax.text(j + 0.5, i + 0.18, labels[i][j],
                    ha="center", va="center", fontsize=9,
                    color=color, alpha=0.6, fontstyle="italic")


def plot_metrics(ax: plt.Axes, y_true, y_pred, classes) -> None:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    metrics = {"Precision": prec * 100, "Recall": rec * 100, "F1-Score": f1 * 100}
    colors  = [C_CATS, C_DOGS]
    x       = np.arange(len(metrics))
    w       = 0.32

    for ci, (cls, color) in enumerate(zip(classes, colors)):
        vals = [v[ci] for v in metrics.values()]
        offset = (ci - 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=cls.capitalize(),
                      color=color, alpha=0.88, zorder=3,
                      linewidth=0, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=C_DARK)

    ax.set_ylim(88, 100.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics.keys(), fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Métricas por Classe", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_roc(ax: plt.Axes, y_true, y_prob) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=C_CATS, lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})", zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_CATS)
    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1.5, label="Aleatório (AUC = 0.50)")

    # Ponto ótimo (máx Youden)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], s=90, color=C_DOGS, zorder=5,
               label=f"Ótimo  (FPR={fpr[best_idx]:.2f}, TPR={tpr[best_idx]:.2f})")

    ax.set_xlabel("Taxa Falso Positivo (FPR)", fontsize=12, labelpad=8)
    ax.set_ylabel("Taxa Verdadeiro Positivo (TPR)", fontsize=12, labelpad=8)
    ax.set_title("Curva ROC", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    ax.yaxis.grid(True, alpha=0.35); ax.xaxis.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def plot_confidence(ax: plt.Axes, y_true, y_prob) -> None:
    bins = np.linspace(0, 1, 31)

    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.70, color=C_CATS,
            label="Gato (real)", density=True, zorder=3)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.70, color=C_DOGS,
            label="Cachorro (real)", density=True, zorder=3)
    ax.axvline(0.5, color=C_DARK, linestyle="--", linewidth=1.8,
               label="Limiar (0.5)", zorder=4)

    # Anotação zona de dúvida
    ax.axvspan(0.4, 0.6, alpha=0.08, color=C_AMBER, zorder=2)
    ax.text(0.5, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 5,
            "zona\nde dúvida", ha="center", va="center",
            fontsize=8, color=C_AMBER, fontstyle="italic")

    conf = np.maximum(y_prob, 1 - y_prob)
    high = (conf >= 0.8).mean() * 100
    ax.text(0.02, 0.93, f"Alta confiança\n(≥0.8): {high:.1f}%",
            transform=ax.transAxes, fontsize=9, color=C_GREEN,
            va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREEN, alpha=0.8))

    ax.set_xlabel("Probabilidade predita  P(cachorro)", fontsize=12, labelpad=8)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuição de Confiança", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Dispositivo: {DEVICE}")
    print("Coletando predições...", flush=True)
    y_true, y_pred, y_prob = collect()

    classes = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms).classes
    print(f"Classes: {classes}  |  {len(y_true)} imgs\n")

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    acc        = (y_true == y_pred).mean() * 100
    high_conf  = (np.maximum(y_prob, 1 - y_prob) >= 0.8).mean() * 100
    _, _, roc_auc_val = roc_curve(y_true, y_prob)
    roc_auc_val = auc(*roc_curve(y_true, y_prob)[:2])

    # ── Layout ────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    fig.suptitle("CatDogNet v2  —  Dashboard de Avaliação",
                 fontsize=17, fontweight="bold", color=C_DARK, y=0.98)

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 4], hspace=0.28)

    # Faixa KPI
    kpi_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.18)
    kpi_data = [
        (f"{acc:.2f}%",       "Acurácia Geral",         C_DARK),
        (f"{f1[0]*100:.2f}%", "F1-Score  Cats",         C_CATS),
        (f"{f1[1]*100:.2f}%", "F1-Score  Dogs",         C_DOGS),
        (f"{high_conf:.1f}%", "Alta Confiança (≥ 0.8)", C_GREEN),
    ]
    for i, (val, lbl, col) in enumerate(kpi_data):
        ax = fig.add_subplot(kpi_gs[i])
        kpi_card(ax, val, lbl, col)

    # 4 gráficos
    chart_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.38)
    ax_cm   = fig.add_subplot(chart_gs[0])
    ax_met  = fig.add_subplot(chart_gs[1])
    ax_roc  = fig.add_subplot(chart_gs[2])
    ax_hist = fig.add_subplot(chart_gs[3])

    plot_confusion(ax_cm,   y_true, y_pred, classes)
    plot_metrics(ax_met,    y_true, y_pred, classes)
    plot_roc(ax_roc,        y_true, y_prob)
    plot_confidence(ax_hist, y_true, y_prob)

    plt.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Salvo: {OUTPUT}")


if __name__ == "__main__":
    main()
