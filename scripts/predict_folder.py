import argparse, json
from pathlib import Path

import torch
from torchvision import transforms, datasets
import timm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def get_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(model_name):
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    img_size = 299 if "xception" in model_name.lower() else 224
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def save_confusion_matrix(y_true, y_pred, classes, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4,4), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


@torch.no_grad()
def main(args):
    device = get_device()
    root = Path(args.folder)
    tfm = build_transform(args.model)

    # If folder has class subdirs (real/fake), use ImageFolder; else, make a dummy dataset
    has_labels = (root / "real").exists() and (root / "fake").exists()
    if has_labels:
        ds = datasets.ImageFolder(root, transform=tfm)
        classes = ds.classes
    else:
        # flat folder: create a list of files & map all to label -1
        exts = (".jpg",".jpeg",".png",".bmp",".webp")
        files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        classes = ["fake","real"]
        class Flat(torch.utils.data.Dataset):
            def __init__(self, files, tfm): self.files, self.tfm = files, tfm
            def __len__(self): return len(self.files)
            def __getitem__(self, i):
                from PIL import Image
                x = self.tfm(Image.open(self.files[i]).convert("RGB"))
                return x, -1, str(self.files[i])
        ds = Flat(files, tfm)

    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                     num_workers=2, pin_memory=torch.cuda.is_available())

    model = timm.create_model(args.model, pretrained=False, num_classes=2)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    preds, trues, paths = [], [], []
    for batch in ld:
        if has_labels:
            xb, yb = batch
            path_batch = None
        else:
            xb, yb, path_batch = batch
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().tolist()
        preds.extend(pred)

        if has_labels:
            trues.extend(yb.cpu().tolist())
        else:
            trues.extend([-1] * len(pred))
            if path_batch is None:
                paths.extend([""] * len(pred))
            else:
                paths.extend(list(path_batch))

    # print quick lines
    label_map = {0: "fake", 1: "real"}
    if paths:
        for pth, pr in zip(paths, preds):
            print(f"{Path(pth).name:40s} -> {label_map[pr]}")

    # metrics only when we have labels
    out_dir = root.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_labels:
        acc = (np.array(trues) == np.array(preds)).mean()
        p, r, f1, _ = precision_recall_fscore_support(trues, preds, labels=[0,1],
                                                      average="binary", pos_label=1)
        macro_f1 = f1_score(trues, preds, average="macro")
        cm_png = out_dir / "cm_predict_folder.png"
        save_confusion_matrix(trues, preds, classes=["fake","real"], out_png=str(cm_png))
        with open(out_dir / "metrics_predict_folder.json", "w") as f:
            json.dump({
                "acc": float(acc),
                "precision_real": float(p),
                "recall_real": float(r),
                "f1_real": float(f1),
                "macro_f1": float(macro_f1)
            }, f, indent=2)
        print(f"[BATCH] acc={acc:.3f}  F1(real)={f1:.3f}  P={p:.3f}  R={r:.3f}  macroF1={macro_f1:.3f}  |  CM→ {cm_png}")
    else:
        print("[BATCH] No labels detected → skipping confusion matrix/F1. Predictions printed above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="resnet50")
    args = ap.parse_args()
    main(args)
