# src/train.py
import argparse, csv, json, time, random, datetime
from pathlib import Path
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# --------------- utils --------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def import_model(class_name: str):
    import importlib
    mdl = importlib.import_mod_
