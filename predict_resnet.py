"""
Valida o modelo ResNet18 com imagens reais.

Modos de uso:
    # Uma imagem específica
    python predict_resnet.py caminho/imagem.jpg

    # Todas imagens do val set (mostra acertos/erros/matriz de confusão)
    python predict_resnet.py --val

    # N imagens aleatórias do val set
    python predict_resnet.py --val --n 20
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_resnet18.pth")
DATA_DIR = Path("dataset/val")
IMG_SIZE = 224

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def predict_single(model: nn.Module, image_path: Path) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    # cats=0, dogs=1 (ordem alfabética ImageFolder)
    label = "cachorro" if prob >= 0.5 else "gato"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence


def run_single(image_path: str) -> None:
    model = load_model()
    label, confidence = predict_single(model, Path(image_path))
    print(f"\nImagem : {image_path}")
    print(f"Predição: {label}  ({confidence:.1%} confiança)")


def run_val(n: int | None) -> None:
    model = load_model()

    # Coleta imagens do val com label real
    samples: list[tuple[Path, str]] = []
    for class_name in ["cats", "dogs"]:
        folder = DATA_DIR / class_name
        real_label = "gato" if class_name == "cats" else "cachorro"
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for img in images:
            samples.append((img, real_label))

    if n:
        random.shuffle(samples)
        samples = samples[:n]
    else:
        samples.sort(key=lambda x: x[0].name)

    # Roda predições
    correct = 0
    errors: list[tuple[str, str, str, float]] = []

    print(f"\n{'Imagem':<30} {'Real':<10} {'Predito':<10} {'Conf':>6}  {'OK?'}")
    print("-" * 70)

    for img_path, real_label in samples:
        pred_label, confidence = predict_single(model, img_path)
        ok = pred_label == real_label
        correct += int(ok)
        flag = "✓" if ok else "✗"

        print(f"{img_path.name:<30} {real_label:<10} {pred_label:<10} {confidence:>5.1%}  {flag}")

        if not ok:
            errors.append((img_path.name, real_label, pred_label, confidence))

    total = len(samples)
    acc = correct / total

    # Matriz de confusão manual
    tp = sum(1 for _, r, p, _ in [("", r, p, c) for _, r, p, c in errors] if False)  # placeholder
    cats_total = sum(1 for _, r in samples if r == "gato")
    dogs_total = sum(1 for _, r in samples if r == "cachorro")
    cats_correct = sum(1 for img, r in samples if r == "gato" and predict_single(model, img)[0] == "gato")
    dogs_correct = sum(1 for img, r in samples if r == "cachorro" and predict_single(model, img)[0] == "cachorro")

    print(f"\n{'─'*70}")
    print(f"Acurácia geral : {correct}/{total} = {acc:.1%}")
    print(f"Gatos corretos : {cats_correct}/{cats_total}")
    print(f"Cachorros corretos: {dogs_correct}/{dogs_total}")

    if errors:
        print(f"\nErros ({len(errors)}):")
        for name, real, pred, conf in errors:
            print(f"  {name}: era {real}, predito {pred} ({conf:.1%})")
    else:
        print("\nNenhum erro!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Caminho da imagem")
    parser.add_argument("--val", action="store_true", help="Roda no val set inteiro")
    parser.add_argument("--n", type=int, default=None, help="N imagens aleatórias do val")
    args = parser.parse_args()

    print(f"Dispositivo: {DEVICE}")

    if args.val or args.n:
        run_val(n=args.n)
    elif args.image:
        run_single(args.image)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
