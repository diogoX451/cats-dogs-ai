"""
Classifica uma imagem nova com o modelo treinado.

Uso:
    python predict.py caminho/para/imagem.jpg
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import CatDogNet, IMG_SIZE, MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image_path: str) -> None:
    model = CatDogNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0).to(DEVICE)   # [1, 3, H, W]

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    # cats=0, dogs=1 (ordem alfabética ImageFolder)
    label = "cachorro" if prob >= 0.5 else "gato"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"Predição: {label}  (confiança: {confidence:.1%})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python predict.py <caminho_imagem>")
        sys.exit(1)
    predict(sys.argv[1])
