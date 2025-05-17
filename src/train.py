# src/train.py
"""
Training script
---------------
$ python src/train.py --data-dir data/see-animals-dataset \
                      --model cnn01 --img-size 128 --batch 32 --epochs 20 \
                      --notes "baseline run"
"""
import argparse, csv, time, random, datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models

# δικά μας modules
from src.dataset import get_dataloaders
from src.models  import CNN01                             # minimal 3-layer CNN

# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# --------------------------- loops ---------------------------------------- #
def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); running = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)

@torch.inference_mode()
def valid_epoch(model, loader, criterion, device):
    model.eval(); loss = correct = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss += criterion(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss / len(loader), correct / n

# ------------------------- build model ------------------------------------ #
def build_model(name: str, n_classes: int, img_size: int, device):
    if name == "cnn01":
        model = CNN01(n_classes, img_size)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model.to(device)

# ------------------------------ main -------------------------------------- #
def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, classes = get_dataloaders(
        cfg.data_dir, cfg.img_size, cfg.batch, cfg.seed
    )

    model = build_model(cfg.model, len(classes), cfg.img_size, device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    out_dir = Path(cfg.output); out_dir.mkdir(exist_ok=True)
    csv_log = out_dir / f"{cfg.model}_stats.csv"
    best_ckpt = out_dir / f"best_{cfg.model}.pt"
    best_val  = float("inf")

    with open(csv_log, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["epoch","train","val","acc"])

        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_dl, criterion, optimizer, device)
            val_loss, val_acc = valid_epoch(model, val_dl, criterion, device)
            scheduler.step(val_loss)

            writer.writerow([ep, tr_loss, val_loss, val_acc])
            print(f"[{ep}/{cfg.epochs}] train {tr_loss:.4f} | "
                  f"val {val_loss:.4f} | acc {val_acc:.3f} "
                  f"({time.time()-t0:.1f}s)")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_ckpt)

    # ---------------- experiment log -------------------------------------- #
    import pandas as pd
    run_row = {
        "run_id"     : datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "date"       : datetime.datetime.now().isoformat(timespec="seconds"),
        "data_dir"   : Path(cfg.data_dir).resolve().as_posix(),
        "model"      : cfg.model,
        "params"     : sum(p.numel() for p in model.parameters()),
        "img_size"   : cfg.img_size,
        "batch"      : cfg.batch,
        "epochs"     : cfg.epochs,
        "lr_init"    : cfg.lr,
        "best_epoch" : ep,
        "val_loss"   : best_val,
        "val_acc"    : val_acc,
        "ckpt_path"  : best_ckpt.resolve().as_posix(),
        "csv_log"    : csv_log.resolve().as_posix(),
        "notes"      : cfg.notes
    }
    exp_file = Path("experiments.csv")
    df = pd.read_csv(exp_file) if exp_file.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([run_row])], ignore_index=True)
    df.to_csv(exp_file, index=False)
    print("✔ Logged run to", exp_file)

# ----------------------------- CLI ---------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/see-animals-dataset")
    p.add_argument("--model", choices=["cnn01","resnet18"], default="cnn01")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default="checkpoints")
    p.add_argument("--notes",    default="", help="free-text notes for the run")
    main(p.parse_args())
