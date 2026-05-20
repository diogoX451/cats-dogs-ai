"""
Divide train/ em train/ e val/ (80/20).
Executa UMA vez. Verifica se val já tem imagens antes de mover.
"""
import os
import shutil
import random
from pathlib import Path

DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
VAL_RATIO = 0.2
SEED = 42


def split_class(class_name: str) -> None:
    src = TRAIN_DIR / class_name
    dst = VAL_DIR / class_name
    dst.mkdir(parents=True, exist_ok=True)

    all_images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))

    # Se val já tem imagens (ignora .gitkeep), não reexecuta
    already_in_val = len(list(dst.glob("*.jpg"))) + len(list(dst.glob("*.png")))
    if already_in_val > 0:
        print(f"[{class_name}] val já tem {already_in_val} imagens. Pulando.")
        return

    random.seed(SEED)
    random.shuffle(all_images)

    n_val = int(len(all_images) * VAL_RATIO)
    val_images = all_images[:n_val]

    for img_path in val_images:
        shutil.move(str(img_path), str(dst / img_path.name))

    print(f"[{class_name}] train: {len(all_images) - n_val}  val: {n_val}")


def main() -> None:
    for class_name in ["cats", "dogs"]:
        split_class(class_name)


if __name__ == "__main__":
    main()
