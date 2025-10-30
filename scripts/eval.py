import argparse, time, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from contextlib import nullcontext

import torchvision as tv
from torchvision import transforms
import timm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

try:
    from torch.cuda.amp import GradScaler
except Exception:
    GradScaler = None


def get_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(img_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_loaders(data_dir, img_size, batch_size, num_workers=2):
    train_tf, val_tf = build_transforms(img_size)
    train_ds = tv.datasets.ImageFolder(Path(data_dir) / "Train", transform=train_tf)
    val_ds   = tv.datasets.ImageFolder(Path(data_dir) / "val",   transform=val_tf)
    pin = torch.cuda.is_available()
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return train_ds, val_ds, train_ld, val_ld


def build_model(model_name, num_classes, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def save_confusion_matrix(y_true, y_pred, classes, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    y_true, y_pred = [], []
    use_cuda_amp = (device.type == "cuda")
    amp_ctx = (autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext())
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with amp_ctx:
            logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    acc = (correct / total) if total else 0.0
    return acc, y_true, y_pred


def train(args):
    device = get_device()
    print(f"Device: {device}")

    img_size = 299 if "xception" in args.model.lower() else 224
    train_ds, val_ds, train_ld, val_ld = build_loaders(args.data_dir, img_size, args.batch_size)

    classes = train_ds.classes
    print(f"Classes: {classes} (n={len(classes)})")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    model = build_model(args.model, num_classes=len(classes), pretrained=True).to(device)
    torch.set_float32_matmul_precision("high")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    patience_left = args.patience
    ckpt_dir = Path(args.checkpoints); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model.replace('/', '_')}_best.pth"
    out_dir = ckpt_dir.parent / "outputs"; out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda_amp = (device.type == "cuda")
    scaler = GradScaler() if (use_cuda_amp and GradScaler is not None) else None
    amp_ctx = (autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext())

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(xb)
                loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_ds))

        # ---- VALIDATION + METRICS ----
        val_acc, y_true, y_pred = evaluate(model, val_ld, device)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1],
                                                      average="binary", pos_label=1)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        cm_png = out_dir / f"cm_epoch_{epoch:02d}.png"
        save_confusion_matrix(y_true, y_pred, classes, str(cm_png))
        with open(out_dir / f"metrics_epoch_{epoch:02d}.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "val_acc": float(val_acc),
                "precision_real": float(p),
                "recall_real": float(r),
                "f1_real": float(f1),
                "macro_f1": float(macro_f1)
            }, f, indent=2)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f} "
              f"| F1(real)={f1:.3f} P={p:.3f} R={r:.3f} macroF1={macro_f1:.3f} | time={dt:.1f}s "
              f"| CM→ {cm_png.name}")

        # early stopping on val_acc
        if val_acc > best_acc:
            best_acc = val_acc
            patience_left = args.patience
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ New best acc {best_acc:.4f}. Saved to {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    print(f"Best val_acc: {best_acc:.4f}")
    print(f"Checkpoint: {ckpt_path if ckpt_path.exists() else 'not saved'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier on dataset")
    p.add_argument("--data-dir", type=str, default="../data/subsets/18k")
    p.add_argument("--checkpoints", type=str, default="../checkpoints")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
