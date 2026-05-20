"""
Transfer Learning com ResNet18 para gato vs cachorro.

Estratégia em 2 fases:
  Fase 1 — Feature Extraction (5 epochs)
    Backbone ResNet18 congelado (pesos ImageNet intocados)
    Treina só o classificador final (512 → 1)
    Rápido, converge logo, ~90%+ em poucas epochs

  Fase 2 — Fine-tuning (15 epochs)
    Descongela TODO o backbone
    LR muito menor (1e-5) para não destruir features ImageNet
    Ajuste fino para o domínio específico (gatos/cachorros)
    Empurra para ~95-99%

Por que ResNet18 é melhor que CNN do zero aqui:
  ImageNet tem 1.2M imagens e 1000 classes.
  ResNet18 já aprendeu detectar bordas, texturas, formas, olhos, orelhas, pelos.
  Nós só ensinamos "isso é gato, isso é cachorro" em cima dessas features prontas.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ─── Hiperparâmetros ──────────────────────────────────────────────────────────
IMG_SIZE = 224          # ResNet18 foi treinada com 224x224 — mantém esse tamanho
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5       # feature extraction (backbone congelado)
EPOCHS_PHASE2 = 15      # fine-tuning (backbone descongelado)
LR_PHASE1 = 1e-3        # LR alto: só o classificador novo aprende
LR_PHASE2 = 1e-5        # LR baixo: ajuste fino sem destruir features ImageNet
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("dataset")
MODEL_PATH = Path("cats_dogs_resnet18.pth")


# ─── Transforms ───────────────────────────────────────────────────────────────
# Normalização IGUAL ao ImageNet — obrigatório quando usa pesos pré-treinados
# ResNet18 foi normalizada com esses valores; usar outros = features erradas
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Modelo ───────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    # Carrega ResNet18 com pesos ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Congela TODOS os parâmetros do backbone
    # → gradiente não flui para eles na fase 1
    for param in model.parameters():
        param.requires_grad = False

    # Substitui classificador final: 512 features → 1 saída binária
    # Este é o ÚNICO layer que treina na fase 1
    # requires_grad=True por padrão em novos layers
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model.to(DEVICE)


def unfreeze_backbone(model: nn.Module) -> None:
    """Descongela todos os parâmetros para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ─── Loop de treino / validação ───────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epochs: int,
    phase_name: str,
    best_val_acc: float,
) -> float:
    print(f"\n{'─'*48}")
    print(f" {phase_name}")
    print(f"{'─'*48}")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>9} {'Val Loss':>8} {'Val Acc':>7}")
    print("-" * 48)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        flag = " ★" if val_acc > best_val_acc else ""
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}"
            f"  {val_loss:>8.4f}  {val_acc:>6.2%}{flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    return best_val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Dispositivo: {DEVICE}")

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transforms)

    print(f"Classes: {train_dataset.classes}")
    print(f"Treino: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model()
    criterion = nn.BCEWithLogitsLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParâmetros treináveis: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # ── Fase 1: Feature Extraction ─────────────────────────────────────────
    # Só o classificador (fc) tem requires_grad=True
    # optimizer recebe apenas esses parâmetros
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, patience=2, factor=0.5)

    best_acc = run_phase(
        model, train_loader, val_loader, criterion,
        optimizer1, scheduler1, EPOCHS_PHASE1,
        "FASE 1 — Feature Extraction (backbone congelado)",
        best_val_acc=0.0,
    )

    # ── Fase 2: Fine-tuning ────────────────────────────────────────────────
    # Descongela backbone inteiro
    # LR muito menor para não destruir features ImageNet aprendidas
    unfreeze_backbone(model)
    optimizer2 = optim.Adam(model.parameters(), lr=LR_PHASE2)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, patience=3, factor=0.5)

    best_acc = run_phase(
        model, train_loader, val_loader, criterion,
        optimizer2, scheduler2, EPOCHS_PHASE2,
        "FASE 2 — Fine-tuning (backbone descongelado, LR=1e-5)",
        best_val_acc=best_acc,
    )

    print(f"\nMelhor val accuracy: {best_acc:.2%}")
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
