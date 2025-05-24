# src/dataset.py
import shutil, random, json
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

# ----------------------------- helpers ---------------------------------- #
def _copy_subset(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        shutil.copy2(fp, dst_dir / fp.name)

def _prepare_split(root: Path,
                   val_pct: float = 0.2,
                   test_pct: float = 0.1,
                   seed: int = 42) -> Path:
    """
    Δημιουργεί train/val/test dirs κάτω από root.parent / (root.name + '_split')
    και αντιγράφει τα αρχεία μία φορά.
    """
    split_root = root.with_name(root.name + "_split")
    if split_root.exists():
        return split_root                     # έγινε ήδη

    random.seed(seed)
    subsets = ["train", "val", "test"]
    for cls_dir in [p for p in root.iterdir() if p.is_dir()]:
        imgs = [p for p in cls_dir.glob("*.jpg")]
        random.shuffle(imgs)

        n_total = len(imgs)
        n_val  = int(n_total * val_pct)
        n_test = int(n_total * test_pct)
        n_train = n_total - n_val - n_test

        split_counts = dict(zip(subsets,
                                [n_train, n_val, n_test]))
        idx = 0
        for subset in subsets:
            part = imgs[idx : idx + split_counts[subset]]
            idx += split_counts[subset]
            dst = split_root / subset / cls_dir.name
            _copy_subset(part, dst)

    return split_root

# ------------------------- main API ------------------------------------ #
_transform_cache = {}   # μην ξαναφτιάχνουμε Transforms για ίδιο img_size


def get_dataloaders(root_dir: str | Path,
                    img_size: int,
                    batch: int,
                    seed: int,
                    val_pct: float = 0.2,
                    test_pct: float = 0.1,
                    with_test: bool = False
                    ) -> Tuple[DataLoader, DataLoader, list] | Tuple[DataLoader, DataLoader, DataLoader, list]:

    root = Path(root_dir).expanduser().resolve()
    split_root = _prepare_split(root, val_pct, test_pct, seed)

    # --- transforms (απλά) ---
    if img_size not in _transform_cache:
        _transform_cache[img_size] = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
    transform = _transform_cache[img_size]

    train_ds = ImageFolder(split_root / "train", transform=transform)
    val_ds   = ImageFolder(split_root / "val",   transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    if with_test:
        test_ds = ImageFolder(split_root / "test", transform=transform)
        test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
        return train_dl, val_dl, test_dl, train_ds.classes

    return train_dl, val_dl, train_ds.classes
