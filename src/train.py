import argparse as ap
from pathlib import Path
import time
import pandas as pd
import torch
from torch import optim, no_grad
from models import get_model
from dataset import get_dataloaders

# =============================
# ✅ EarlyStopping ενσωματωμένο
# =============================
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# =============================
# ✅ Χρήσιμες συναρτήσεις
# =============================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total

def valid_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, correct / total

# =============================
# ✅ Κύρια ροή εκπαίδευσης
# =============================
def main(cfg):
    start_time = time.time()
    dev = get_device()
    print(f"✅ Device: {dev}")

    tr_dl, val_dl = get_dataloaders(
        cfg.data_dir, imgsize=cfg.img, batch_size=cfg.batch
    )

    model = get_model(cfg.model)
    model.to(dev)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = torch.nn.CrossEntropyLoss()
    early = EarlyStopping(patience=cfg.patience)

    for ep in range(cfg.epochs):
        print(f"\nEpoch {ep+1}/{cfg.epochs}")
        tr_loss, tr_acc = train_epoch(model, tr_dl, crit, opt, dev)
        vl_loss, vl_acc = valid_epoch(model, val_dl, crit, dev)

        print(f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f}")
        print(f"Valid Loss: {vl_loss:.4f} | Acc: {vl_acc:.4f}")

        early(vl_loss)
        if early.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    # Save model
    out_dir = Path("checkpoints") / cfg.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.model}_{cfg.lr}_{cfg.batch}.pt"
    torch.save(model.state_dict(), out_path)

    # Save experiment
    total_time = time.time() - start_time
    log_path = Path("experiments.csv")
    run_data = {
        "model": cfg.model,
        "lr": cfg.lr,
        "batch": cfg.batch,
        "epochs": ep+1,
        "train_acc": tr_acc,
        "valid_acc": vl_acc,
        "train_loss": tr_loss,
        "valid_loss": vl_loss,
        "notes": cfg.notes,
        "time": round(total_time, 2)
    }

    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([run_data])], ignore_index=True)
    else:
        df = pd.DataFrame([run_data])

    df.to_csv(log_path, index=False)
    print(f"✅ Results saved to {log_path}")

# =============================
# ✅ Ορισμός παραμέτρων γραμμής εντολών
# =============================
if __name__ == "__main__":
    pa = ap.ArgumentParser()
    pa.add_argument("--model", type=str, required=True)
    pa.add_argument("--data-dir", type=str, required=True)
    pa.add_argument("--img", type=int, default=128)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--lr", type=float, default=0.001)
    pa.add_argument("--patience", type=int, default=3)
    pa.add_argument("--notes", type=str, default="")
    main(pa.parse_args())
