"""
Verifica estabilidade do modelo CatDogNet: rodar inferência 3x em todas
as imagens do val set e checar se a predição é idêntica nas 3 rodadas.

Com model.eval() + torch.no_grad() + seeds fixas, o resultado deve ser
100% determinístico. Qualquer instabilidade indica bug de aleatoriedade.

Uso:
    python3 stability_check.py
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
from torchvision import datasets, transforms

from train import CatDogNet, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_scratch.pth")
DATA_DIR = Path("dataset/val")
RUNS = 3


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
def predict_all(model: CatDogNet, loader: torch.utils.data.DataLoader) -> np.ndarray:
    preds = []
    for images, _ in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)
        batch_preds = (probs >= 0.5).long().cpu().tolist()
        preds.extend(batch_preds)
    return np.array(preds)


def run_stability_check() -> None:
    print(f"Dispositivo : {DEVICE}")
    print(f"Modelo      : {MODEL_PATH}")
    print(f"Val dir     : {DATA_DIR}")
    print(f"Rodadas     : {RUNS}\n")

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    classes = dataset.classes
    print(f"Classes: {classes}")
    print(f"Total imagens: {len(dataset)}\n")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    model = load_model()

    all_preds = []
    for run in range(1, RUNS + 1):
        print(f"Rodada {run}/{RUNS}...", end=" ", flush=True)
        preds = predict_all(model, loader)
        all_preds.append(preds)
        print(f"classes únicas: {np.unique(preds, return_counts=True)}")

    all_preds = np.stack(all_preds)  # [RUNS, N]

    # Verificar consistência: todas as runs devem ser idênticas
    stable_mask = np.all(all_preds == all_preds[0], axis=0)
    unstable_count = int((~stable_mask).sum())
    total = len(stable_mask)
    stable_pct = stable_mask.mean()

    print(f"\n{'─'*50}")
    print(f"Imagens estáveis  : {total - unstable_count:,}/{total:,}  ({stable_pct:.4%})")
    print(f"Imagens instáveis : {unstable_count:,}/{total:,}  ({1-stable_pct:.4%})")

    if unstable_count == 0:
        print("\n✓ Modelo 100% determinístico — nenhuma predição variou entre rodadas.")
    else:
        print(f"\n✗ {unstable_count} imagens tiveram predições inconsistentes!")
        unstable_indices = np.where(~stable_mask)[0]
        print("Índices instáveis (primeiros 20):", unstable_indices[:20].tolist())
        print("Predições por rodada (primeiros 5 instáveis):")
        for idx in unstable_indices[:5]:
            img_path, label = dataset.samples[idx]
            row = all_preds[:, idx].tolist()
            print(f"  idx={idx}  real={classes[label]}  preds={row}  file={Path(img_path).name}")

    # Distribuição de confiança (prob) — rodar uma vez para estatísticas
    print("\nDistribuição de confiança (rodada 1):")
    probs_all = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            probs_all.extend(probs)
    probs_all = np.array(probs_all)

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(probs_all, bins=bins)
    print(f"{'Faixa prob':>14} {'Qtd':>6} {'%':>6}")
    for i, c in enumerate(counts):
        print(f"  {bins[i]:.1f} – {bins[i+1]:.1f}   {c:>6,}  {c/total:>5.1%}")

    high_conf = ((probs_all < 0.2) | (probs_all > 0.8)).sum()
    print(f"\nAlta confiança (prob<0.2 ou >0.8): {high_conf:,}/{total:,} = {high_conf/total:.1%}")
    low_conf = ((probs_all >= 0.4) & (probs_all <= 0.6)).sum()
    print(f"Baixa confiança (0.4–0.6)        : {low_conf:,}/{total:,} = {low_conf/total:.1%}")


if __name__ == "__main__":
    run_stability_check()
