# scripts/split_kaggle.py
import argparse, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def gather_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

def split_and_copy(src_real: Path, src_fake: Path, out_root: Path, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    real_imgs = gather_images(src_real)
    fake_imgs = gather_images(src_fake)

    assert real_imgs, f"No images found in {src_real}"
    assert fake_imgs, f"No images found in {src_fake}"

    def split_list(files):
        random.shuffle(files)
        n_val = int(len(files) * val_ratio)
        return files[n_val:], files[:n_val]  # train, val

    real_train, real_val = split_list(real_imgs)
    fake_train, fake_val = split_list(fake_imgs)

    def copy_list(files, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            dst = dest / f"{src.stem}_{i}{src.suffix.lower()}"
            shutil.copy2(src, dst)

    # Layout: out_root/{train,val}/{real,fake}
    for split, r_files, f_files in [
        ("train", real_train, fake_train),
        ("val",   real_val,   fake_val),
    ]:
        copy_list(r_files, out_root / split / "real")
        copy_list(f_files, out_root / split / "fake")

    print("✅ Done.")
    print(f"Train: real={len(real_train)}, fake={len(fake_train)}")
    print(f"Val:   real={len(real_val)},   fake={len(fake_val)}")
    print(f"Output at: {out_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to kaggle dataset root containing real/ and fake/")
    ap.add_argument("--out", required=True, help="output dataset root (will create train/ and val/)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    split_and_copy(src/"real", src/"fake", out, args.val_ratio, args.seed)

if __name__ == "__main__":
    main()
