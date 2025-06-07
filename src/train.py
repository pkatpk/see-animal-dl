# src/train.py
"""
Dynamic trainer:
$ python src/train.py --model CNN01 --epochs 10 --notes "baseline"
"""
import argparse, csv, json, time, random, datetime
from pathlib import Path
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


# --------------- utils --------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def import_model(class_name: str):
    import importlib
    mdl = importlib.import_module("src.models")
    try:       return getattr(mdl, class_name)
    except AttributeError:
        raise ValueError(f"Model '{class_name}' not found in src.models")

# --------------- epoch loops --------------------------------------------- #
def train_epoch(model, loader, crit, opt, dev):
    model.train(); loss_sum=acc_sum=n=0
    for X,y in loader:
        X,y = X.to(dev), y.to(dev)
        opt.zero_grad(); out = model(X); loss = crit(out,y)
        loss.backward(); opt.step()
        loss_sum += loss.item()
        acc_sum  += (out.argmax(1)==y).sum().item()
        n += y.size(0)
    return loss_sum/len(loader), acc_sum/n

@torch.inference_mode()
def valid_epoch(model, loader, crit, dev):
    model.eval(); loss_sum=acc_sum=n=0
    for X,y in loader:
        X,y = X.to(dev), y.to(dev)
        out = model(X); loss_sum += crit(out,y).item()
        acc_sum += (out.argmax(1)==y).sum().item(); n += y.size(0)
    return loss_sum/len(loader), acc_sum/n

# --------------- main ---------------------------------------------------- #
def main(cfg):
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.dataset import get_dataloaders
    tr_dl, val_dl, classes = get_dataloaders(
        cfg.data_dir, cfg.img, cfg.batch, cfg.seed)

    Model = import_model(cfg.model)
    kw = json.loads(cfg.model_kw) if cfg.model_kw else {}
    model = Model(len(classes), cfg.img, **kw).to(dev)

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',3,0.5)

    out = Path(cfg.output); out.mkdir(exist_ok=True)
    csv_log = out / f"{cfg.model}_stats.csv"
    best_pt = out / f"best_{cfg.model}.pt"
    best = float("inf")

    tr_L, val_L, tr_A, val_A = [], [], [], []

    with open(csv_log,"w",newline="") as f:
        wr = csv.writer(f); wr.writerow(["epoch","tr_loss","val_loss","tr_acc","val_acc"])
        for ep in range(1, cfg.epochs+1):
            t0=time.time()
            tr_loss, tr_acc = train_epoch(model,tr_dl,crit,opt,dev)
            vl_loss, vl_acc = valid_epoch(model,val_dl,crit,dev)
            tr_L.append(tr_loss); val_L.append(vl_loss)
            tr_A.append(tr_acc);  val_A.append(vl_acc)
            sch.step(vl_loss)
            wr.writerow([ep,tr_loss,vl_loss,tr_acc,vl_acc])
            print(f"[{ep}/{cfg.epochs}] tr {tr_loss:.4f}/{tr_acc:.3f} | "
                  f"vl {vl_loss:.4f}/{vl_acc:.3f}  ({time.time()-t0:.1f}s)")
            if vl_loss<best:
                best=vl_loss; torch.save(model.state_dict(), best_pt)

    # ----- save learning curves ------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(tr_L,label="train loss"); ax1.plot(val_L,label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(tr_A,"--",label="train acc"); ax2.plot(val_A,"--",label="val acc")
    ax2.set_ylabel("accuracy"); ax2.legend(loc="upper right")
    plt.title(cfg.model); plt.tight_layout()
    curve_png = out / f"{cfg.model}_curve.png"
    plt.savefig(curve_png,dpi=120); plt.close()
    print("📈 saved curves to", curve_png)

    # -------- experiment log -------------------------------------------- #
    import pandas as pd
    row = dict(run_id=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
               model=cfg.model, kw=cfg.model_kw,
               img=cfg.img, batch=cfg.batch, epochs=cfg.epochs,
               val_loss=best, val_acc=val_A[-1],
               ckpt=best_pt.as_posix(), curve=curve_png.as_posix(),
               notes=cfg.notes)
    exp = Path("experiments.csv")
    df=pd.read_csv(exp) if exp.exists() else pd.DataFrame()
    df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
    df.to_csv(exp,index=False); print("✔ logged to experiments.csv")

# --------------- CLI ----------------------------------------------------- #
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir",default="data/see-animals-dataset")
    pa.add_argument("--model",required=True,
                    help="class name in src.models, e.g. CNN01")
    pa.add_argument("--model-kw",default="",
                    help='extra kwargs JSON, e.g. \'{"drop":0.3}\'')
    pa.add_argument("--img",type=int,default=128)
    pa.add_argument("--batch",type=int,default=32)
    pa.add_argument("--epochs",type=int,default=20)
    pa.add_argument("--lr",type=float,default=1e-3)
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--output",default="checkpoints")
    pa.add_argument("--notes",default="")
    main(pa.parse_args())
