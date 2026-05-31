"""
Dashboard visual completo do CatDogNet v2.

Painéis:
  1. Matriz de confusão normalizada (%)
  2. Métricas por classe (precision / recall / F1)
  3. Distribuição de probabilidade (histograma)
  4. Acurácia por faixa de confiança

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
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision import datasets, transforms

from train import CatDogNet, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_scratch.pth")
DATA_DIR = Path("dataset/val")
OUTPUT = Path("results_dashboard.png")

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
def collect_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    model = load_model()

    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        probs = torch.sigmoid(model(images)).squeeze(1)
        preds = (probs >= 0.5).long()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, format="%.0f%%", fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    acc = (y_true == y_pred).mean()
    ax.set_title(f"Matriz de Confusão\nAcurácia: {acc:.2%}", fontsize=12, fontweight="bold")

    thresh = 50.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=12, color=color, fontweight="bold")


def plot_metrics(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])

    x = np.arange(len(classes))
    w = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars_p = ax.bar(x - w, precision * 100, w, label="Precision", color=colors[0], alpha=0.85)
    bars_r = ax.bar(x,     recall    * 100, w, label="Recall",    color=colors[1], alpha=0.85)
    bars_f = ax.bar(x + w, f1        * 100, w, label="F1-Score",  color=colors[2], alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(88, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Métricas por Classe", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)


def plot_confidence_hist(ax: plt.Axes, y_prob: np.ndarray, y_true: np.ndarray) -> None:
    cats_probs = y_prob[y_true == 0]
    dogs_probs = y_prob[y_true == 1]

    bins = np.linspace(0, 1, 26)
    ax.hist(cats_probs, bins=bins, alpha=0.65, color="#2196F3", label="cats (real)", density=True)
    ax.hist(dogs_probs, bins=bins, alpha=0.65, color="#FF5722", label="dogs (real)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="limiar 0.5")

    ax.set_xlabel("Probabilidade predita (P(dog))", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.set_title("Distribuição de Confiança", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)


def plot_acc_by_confidence(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    # Confiança = distância ao limiar: max(prob, 1-prob)
    confidence = np.maximum(y_prob, 1 - y_prob)
    thresholds = np.arange(0.5, 1.0, 0.05)
    accs, counts = [], []

    for thr in thresholds:
        mask = confidence >= thr
        if mask.sum() == 0:
            accs.append(np.nan)
            counts.append(0)
        else:
            accs.append((y_true[mask] == y_pred[mask]).mean() * 100)
            counts.append(mask.sum())

    color_line = "#1565C0"
    ax.plot(thresholds, accs, "o-", color=color_line, linewidth=2, markersize=6, label="Acurácia")
    ax.fill_between(thresholds, accs, alpha=0.15, color=color_line)

    ax2 = ax.twinx()
    ax2.bar(thresholds, counts, width=0.04, alpha=0.25, color="#9E9E9E", label="N imagens")
    ax2.set_ylabel("N imagens cobertas", fontsize=10, color="#9E9E9E")
    ax2.tick_params(axis="y", labelcolor="#9E9E9E")

    ax.set_xlabel("Limiar de confiança mínima", fontsize=11)
    ax.set_ylabel("Acurácia (%)", fontsize=11)
    ax.set_title("Acurácia × Confiança", fontsize=12, fontweight="bold")
    ax.set_ylim(92, 101)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=9)


def main() -> None:
    print(f"Dispositivo: {DEVICE}")
    print("Coletando predições...", flush=True)
    y_true, y_pred, y_prob = collect_predictions()

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    classes = dataset.classes
    print(f"Classes: {classes}  |  Total: {len(y_true)} imgs")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("CatDogNet v2 — Dashboard de Avaliação", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_confusion(ax1, y_true, y_pred, classes)
    plot_metrics(ax2, y_true, y_pred, classes)
    plot_confidence_hist(ax3, y_prob, y_true)
    plot_acc_by_confidence(ax4, y_true, y_pred, y_prob)

    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Dashboard salvo: {OUTPUT}")


if __name__ == "__main__":
    main()
