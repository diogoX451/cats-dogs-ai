"""
CNN do zero para classificação gato vs cachorro.

Arquitetura: CatDogNet v2
  4 blocos Conv → MaxPool → BatchNorm → ReLU
  Global Average Pooling no lugar de Flatten
  Classificador FC pequeno com Dropout forte

Mudanças v2 vs v1 e por quê:
  GAP (AdaptiveAvgPool2d)
    v1 usava Flatten → 32768 features → Linear gigante (8.4M params)
    GAP colapsa mapa espacial [B,C,H,W] → [B,C] por média
    Resultado: 256 features em vez de 32768 → 290k params
    Menos params = menos overfitting com dataset pequeno

  4º bloco Conv (256 filtros)
    Compensar profundidade perdida ao reduzir largura
    Features mais abstratas com menos params totais

  Weight decay no optimizer
    Penaliza pesos grandes → regularização L2 explícita
    Complementa Dropout

  Augmentation mais agressivo
    RandomGrayscale: força invariância a cor
    RandomErasing: cobre parte da imagem → modelo não depende de região específica

  CosineAnnealingLR
    LR decai suavemente de 1e-3 → ~0 ao longo dos epochs
    Melhor convergência final que ReduceLROnPlateau para datasets pequenos

  Conv2d      → detecta padrões locais (bordas, texturas, formas)
  MaxPool2d   → reduz resolução, mantém features mais fortes
  BatchNorm2d → normaliza ativações → treino mais estável e rápido
  ReLU        → não-linearidade (sem ela, N camadas = 1 camada linear)
  Dropout     → desliga neurônios aleatórios → força generalização
  GAP         → resume feature maps em vetor compacto
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─── Hiperparâmetros ──────────────────────────────────────────────────────────
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50            # mais epochs: GAP + weight decay resistem a overfitting longo
LR = 1e-3
WEIGHT_DECAY = 1e-4    # L2 regularização
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("cats_dogs_scratch.pth")

# Kaggle: PetImages/Cat + PetImages/Dog sem split
# Local:  dataset/train/cats + dataset/val/cats (já splitado)
KAGGLE_DATA = Path("/kaggle/input/cats-and-dogs-classification-dataset/PetImages")
LOCAL_DATA  = Path("dataset")
ON_KAGGLE   = KAGGLE_DATA.exists()


# ─── Data Augmentation ────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),     # força invariância a cor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # cobre região aleatória
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Arquitetura da CNN v2 ────────────────────────────────────────────────────
class CatDogNet(nn.Module):
    """
    4 blocos convolucionais + Global Average Pooling + FC pequeno.

    Fluxo de dados (batch de 32 imagens 128x128 RGB):
      entrada:  [32, 3,   128, 128]
      bloco1:   [32, 32,  64,  64]
      bloco2:   [32, 64,  32,  32]
      bloco3:   [32, 128, 16,  16]
      bloco4:   [32, 256,  8,   8]
      GAP:      [32, 256]          ← colapsa 8x8 → escalar por canal
      fc:       [32, 1]
      params:   ~290k (vs 8.4M na v1)
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 128→64

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 64→32

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 32→16

            # Bloco 4: mais profundidade sem explodir params (GAP vai comprimir)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # 16→8
        )

        # GAP: [B, 256, 8, 8] → [B, 256, 1, 1] → [B, 256]
        # Cada canal vira UM número (média de toda a feature map)
        # Força o canal a ser globalmente relevante, não dependente de posição
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Dispositivo: {DEVICE}")
    print(f"Ambiente: {'Kaggle' if ON_KAGGLE else 'Local'}")

    if ON_KAGGLE:
        from torch.utils.data import Subset
        # Dois ImageFolder com transforms diferentes, índices não-sobrepostos
        train_base = datasets.ImageFolder(str(KAGGLE_DATA), transform=train_transforms)
        val_base   = datasets.ImageFolder(str(KAGGLE_DATA), transform=val_transforms)
        n = len(train_base)
        val_size = int(0.2 * n)
        torch.manual_seed(42)
        idx = torch.randperm(n).tolist()
        train_dataset: datasets.ImageFolder | Subset = Subset(train_base, idx[val_size:])
        val_dataset:   datasets.ImageFolder | Subset = Subset(val_base,   idx[:val_size])
        print(f"Classes: {train_base.classes}")
    else:
        train_dataset = datasets.ImageFolder(str(LOCAL_DATA / "train"), transform=train_transforms)
        val_dataset   = datasets.ImageFolder(str(LOCAL_DATA / "val"),   transform=val_transforms)
        print(f"Classes: {train_dataset.classes}")

    print(f"Treino: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CatDogNet().to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {params:,}")

    criterion = nn.BCEWithLogitsLoss()

    # weight_decay = L2 regularização: penaliza pesos grandes
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR: LR decai de LR → 0 ao longo de T_max epochs
    # Treino lento e estável no final → melhor convergência
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>9} {'Val Loss':>8} {'Val Acc':>7} {'LR':>8}")
    print("-" * 58)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        flag = " ★" if val_acc > best_val_acc else ""
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}"
            f"  {val_loss:>8.4f}  {val_acc:>6.2%}  {current_lr:.2e}{flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nMelhor val accuracy: {best_val_acc:.2%}")
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
