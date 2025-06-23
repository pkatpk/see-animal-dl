import argparse, time
from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from models import CNN01, CNN02, ResNetTransfer

MODELS = {
    "CNN01": CNN01,
    "CNN02": CNN02,
    "ResNetTransfer": ResNetTransfer
}

def load_model(model_name, checkpoint_path, num_classes, device):
    model = MODELS[model_name](num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def evaluate(model, dataloader, device, class_names):
    y_true, y_pred = [], []
    total, correct, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss_sum += loss.item() * len(yb)

            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += len(yb)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = correct / total
    avg_loss = loss_sum / total

    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"🔍 Test Loss    : {avg_loss:.4f}\n")

    print("📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Saved confusion_matrix.png")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # Load test dataloader only
    _, _, test_dl, class_names = get_dataloaders(
        root_dir=args.data_dir,
        img_size=args.img,
        batch=args.batch,
        with_test=True
    )

    model = load_model(args.model, args.checkpoint, len(class_names), device)

    start = time.time()
    evaluate(model, test_dl, device, class_names)
    print(f"⏱ Done in {(time.time() - start):.1f} sec.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (CNN01, CNN02, ResNetTransfer)")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pt model")
    parser.add_argument("--data-dir", required=True, help="Dataset directory (with split)")
    parser.add_argument("--img", type=int, default=128, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for test")
    args = parser.parse_args()

    main(args)
