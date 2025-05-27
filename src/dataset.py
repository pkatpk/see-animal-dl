"""
src/dataset.py
==============

•  Φορτώνει εικόνες από φακέλους–κλάσεις (PyTorch ImageFolder).  
•  Αν *ΔΕΝ* υπάρχει έτοιμο split, δημιουργεί αυτόματα:

    <root>_split/
        ├─ train/<class>/xxx.jpg
        ├─ val  /<class>/yyy.jpg
        └─ test /<class>/zzz.jpg

•  Παρέχει τη συνάρτηση  get_dataloaders(...)
•  Μπορεί να εκτελεστεί ως **stand-alone** script:

    python -m src.dataset --root data/see_animals_dataset \
                          --val 0.2 --test 0.1 --seed 42
"""

from __future__ import annotations
import random, shutil, argparse
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

# -------------------------- helper utils --------------------------------- #
def _copy_subset(files: List[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for fp in files:
        shutil.copy2(fp, dst / fp.name)

def _prepare_split(
    root: Path,
    val_pct: float,
    test_pct: float,
    seed: int
) -> Path:
    """
    Δημιουργεί τον φάκελο <root>_split αν δεν υπάρχει
    και αντιγράφει εικόνες σε train/val/test υποφακέλους.
    """
    split_root = root.with_name(root.name + "_split")
    if split_root.exists():
        return split_root      # split έχει ήδη γίνει

    rng = random.Random(seed)

    for cls_dir in [p for p in root.iterdir() if p.is_dir()]:
        images = sorted([p for p in cls_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        rng.shuffle(images)

        n_total = len(images)
        n_val   = int(n_total * val_pct)
        n_test  = int(n_total * test_pct)
        n_train = n_total - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train:n_train+n_val],
            "test" : images[n_train+n_val:]
        }

        for subset, files in splits.items():
            _copy_subset(files, split_root / subset / cls_dir.name)

    return split_root

# -------------------------- transforms cache ----------------------------- #
_transform_cache: dict[int, T.Compose] = {}
def _get_transform(img_size: int) -> T.Compose:
    if img_size not in _transform_cache:
        _transform_cache[img_size] = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
    return _transform_cache[img_size]

# -------------------------- public API ----------------------------------- #
def get_dataloaders(
    root_dir: str | Path,
    img_size: int,
    batch: int,
    seed: int = 42,
    val_pct: float = 0.2,
    test_pct: float = 0.1,
    with_test: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]] | Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Επιστρέφει train & val DataLoaders (+ test αν with_test=True).

    Αν το split δεν υπάρχει, δημιουργείται αυτόματα.
    """
    root_dir = Path(root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset folder '{root_dir}' not found")

    split_root = _prepare_split(root_dir, val_pct, test_pct, seed)

    transform = _get_transform(img_size)

    train_ds = ImageFolder(split_root / "train", transform=transform)
    val_ds   = ImageFolder(split_root / "val",   transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    if with_test:
        test_ds = ImageFolder(split_root / "test", transform=transform)
        test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
        return train_dl, val_dl, test_dl, train_ds.classes

    return train_dl, val_dl, train_ds.classes

# -------------------------- CLI entry-point ------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or load train/val/test split and show stats")
    parser.add_argument("--root", required=True,
                        help="Φάκελος με αρχικό dataset (κλάσεις ως υποφακέλους)")
    parser.add_argument("--val",  type=float, default=0.2,
                        help="Ποσοστό validation (default 0.2)")
    parser.add_argument("--test", type=float, default=0.1,
                        help="Ποσοστό test (default 0.1)")
    parser.add_argument("--seed", type=int,   default=42,
                        help="Seed για reproducible split")
    args = parser.parse_args()

    root = Path(args.root)
    split_root = _prepare_split(root, args.val, args.test, args.seed)

    n_train = sum(1 for _ in (split_root / "train").rglob("*.*"))
    n_val   = sum(1 for _ in (split_root / "val").rglob("*.*"))
    n_test  = sum(1 for _ in (split_root / "test").rglob("*.*"))

    print(f"✅ Split ready at {split_root}")
    print(f"   Train: {n_train} images")
    print(f"   Val  : {n_val} images")
    print(f"   Test : {n_test} images")
