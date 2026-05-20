"""
Reorganiza PetImages/ → dataset/train/ e dataset/val/ (80/20).

PetImages tem imagens corrompidas — valida cada arquivo antes de mover.
Executa UMA vez (verifica se já foi feito).
"""

import shutil
import random
from pathlib import Path
from PIL import Image

SRC_CATS = Path("kaggle_raw/PetImages/Cat")
SRC_DOGS = Path("kaggle_raw/PetImages/Dog")
DATASET_DIR = Path("dataset")
VAL_RATIO = 0.2
SEED = 42


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def setup_class(src: Path, class_name: str) -> None:
    train_dst = DATASET_DIR / "train" / class_name
    val_dst = DATASET_DIR / "val" / class_name
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    # Verifica se já foi feito (ignora .gitkeep)
    existing = len(list(train_dst.glob("*.jpg")))
    if existing > 100:
        print(f"[{class_name}] já configurado ({existing} imagens em train). Pulando.")
        return

    print(f"[{class_name}] validando imagens...", flush=True)
    all_images = sorted(src.glob("*.jpg"))

    valid = [p for p in all_images if is_valid_image(p)]
    invalid = len(all_images) - len(valid)
    print(f"[{class_name}] válidas: {len(valid)}  corrompidas: {invalid}")

    random.seed(SEED)
    random.shuffle(valid)

    n_val = int(len(valid) * VAL_RATIO)
    val_images = set(valid[:n_val])
    train_images = valid[n_val:]

    print(f"[{class_name}] copiando {len(train_images)} train + {n_val} val...", flush=True)

    for img in train_images:
        shutil.copy2(img, train_dst / img.name)

    for img in val_images:
        shutil.copy2(img, val_dst / img.name)

    print(f"[{class_name}] done — train: {len(train_images)}  val: {n_val}")


def main() -> None:
    setup_class(SRC_CATS, "cats")
    setup_class(SRC_DOGS, "dogs")

    train_cats = len(list((DATASET_DIR / "train/cats").glob("*.jpg")))
    train_dogs = len(list((DATASET_DIR / "train/dogs").glob("*.jpg")))
    val_cats = len(list((DATASET_DIR / "val/cats").glob("*.jpg")))
    val_dogs = len(list((DATASET_DIR / "val/dogs").glob("*.jpg")))

    print(f"\nDataset final:")
    print(f"  train: {train_cats} gatos + {train_dogs} cachorros = {train_cats+train_dogs}")
    print(f"  val:   {val_cats} gatos + {val_dogs} cachorros = {val_cats+val_dogs}")


if __name__ == "__main__":
    main()
