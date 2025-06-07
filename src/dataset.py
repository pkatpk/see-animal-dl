"""
src/dataset.py
==============
•  Φορτώνει εικόνες από φακέλους–κλάσεις και επιστρέφει DataLoaders.
•  Αν το root διαθέτει ήδη train / val (και προαιρετικά test) φακέλους
   με εικόνες, τους χρησιμοποιεί όπως είναι.
•  Αν ΟΧΙ, δημιουργεί νέα δομή  <root>_split/  με train/val/test.
"""

from __future__ import annotations
import random, shutil, argparse
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


# ----------------------------------------------------------------------
#  Βοηθητικά
# ----------------------------------------------------------------------
def _copy_subset(files: List[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for fp in files:
        shutil.copy2(fp, dst / fp.name)


# ----------------------------------------------------------------------
#  Δημιουργία ή αναγνώριση split
# ----------------------------------------------------------------------
def _prepare_split(
    root: Path,
    val_pct: float = 0.2,
    test_pct: float = 0.1,
    seed: int = 42
) -> Path:
    """
    • Αν ο `root` έχει ΗΔΗ train/val (και ίσως test) φακέλους με εικόνες,
      επιστρέφει τον ίδιο τον root ― δεν ξανακάνει split.
    • Αλλιώς δημιουργεί <root>_split/ με train/val/test (70/20/10 προεπιλογή).
    """
    # --- 1) Έτοιμο split: χρησιμοποιήσέ το όπως είναι ------------------
    if (root / "train").exists() and (root / "val").exists():
        has_train = any((root / "train").rglob("*.*"))
        has_val   = any((root / "val").rglob("*.*"))
        if has_train and has_val:
            return root.resolve()

    # --- 2) Προϋπάρχον <root>_split -----------------------------------
    split_root = root.with_name(root.name + "_split")
    if split_root.exists():
        return split_root.resolve()

    # --- 3) Δημιούργησε νέο split ------------------------------------
    random.seed(seed)
    for cls_dir in [p for p in root.iterdir() if p.is_dir()]:
        images = sorted([p for p in cls_dir.glob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        random.shuffle(images)

        n = len(images)
        n_val  = int(n * val_pct)
        n_test = int(n * test_pct)
        n_train = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train:n_train + n_val],
            "test" : images[n_train + n_val:]
        }

        for subset, files in splits.items():
            _copy_subset(files, split_root / subset / cls_dir.name)

    return split_root.resolve()


# ----------------------------------------------------------------------
#  Μετασχηματισμοί (cached per img_size)
# ----------------------------------------------------------------------
_transform_cache: dict[int, T.Compose] = {}


def _get_transform(img_size: int) -> T.Compose:
    if img_size not in _transform_cache:
        _transform_cache[img_size] = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
    return _transform_cache[img_size]


# ----------------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------------
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
):
    """
    Επιστρέφει train & val DataLoaders (+ test αν with_test=True).

    • Αν το split υπάρχει ήδη, το χρησιμοποιεί.
    • Αλλιώς δημιουργεί νέο split (<root> → <root>_split).
    """
    root_dir = Path(root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset folder '{root_dir}' not found")

    split_root = _prepare_split(root_dir, val_pct, test_pct, seed)
    transform  = _get_transform(img_size)

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


# ----------------------------------------------------------------------
#  CLI για εφάπαξ split & στατιστικά
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Φάκελος με αρχικό dataset (κλάσεις ως υποφακέλους)")
    parser.add_argument("--val",  type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int,   default=42)
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
