"""
Avalia o modelo CatDogNet no val set completo.
Gera matriz de confusão, métricas por classe e salva confusion_matrix.png.

Uso:
    python3 evaluate.py
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms

from train import CatDogNet, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_scratch.pth")
DATA_DIR = Path("dataset/val")
OUTPUT_IMG = Path("confusion_matrix.png")

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
def run_evaluation() -> None:
    print(f"Dispositivo : {DEVICE}")
    print(f"Modelo      : {MODEL_PATH}")
    print(f"Val dir     : {DATA_DIR}\n")

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    # ImageFolder ordena classes alfabeticamente: cats=0, dogs=1
    classes = dataset.classes
    print(f"Classes: {classes}  (0={classes[0]}, 1={classes[1]})")
    print(f"Total imagens: {len(dataset)}\n")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    model = load_model()

    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Matriz de confusão ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    # layout: linhas=real, colunas=predito
    # cm[0,0]=TN(cats ok) cm[0,1]=FP(cats→dogs) cm[1,0]=FN(dogs→cats) cm[1,1]=TP(dogs ok)

    print("Matriz de Confusão (linhas=real, colunas=predito):")
    print(f"{'':>12} {'pred cats':>12} {'pred dogs':>12}")
    print(f"{'real cats':>12} {cm[0,0]:>12,} {cm[0,1]:>12,}")
    print(f"{'real dogs':>12} {cm[1,0]:>12,} {cm[1,1]:>12,}\n")

    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"TN (cats→cats)  : {tn:,}")
    print(f"FP (cats→dogs)  : {fp:,}")
    print(f"FN (dogs→cats)  : {fn:,}")
    print(f"TP (dogs→dogs)  : {tp:,}\n")

    total = len(y_true)
    acc = (tp + tn) / total
    print(f"Acurácia geral  : {acc:.4%}  ({tp+tn:,}/{total:,})\n")

    print("Métricas por classe:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # ── Salvar imagem ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Predito", fontsize=13)
    ax.set_ylabel("Real", fontsize=13)
    ax.set_title(f"Matriz de Confusão — CatDogNet\nAcurácia: {acc:.2%}", fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"\nImagem salva: {OUTPUT_IMG}")


if __name__ == "__main__":
    run_evaluation()
