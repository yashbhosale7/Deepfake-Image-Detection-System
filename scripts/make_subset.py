# scripts/make_subset.py
import os, glob, random, shutil
from pathlib import Path
from collections import defaultdict

# ========= EDIT ONLY THESE =========
SUBSET_NAME = "18k"       # folder name under data/subsets/
RANDOM_SEED = 42
OVERWRITE   = True        # delete existing subset if it exists

# target sample counts per split/class
# (Use your dataset's exact split names: "Train" and "val")
COUNTS = {
    ("Train", "real"): 8000,
    ("Train", "fake"): 8000,
    ("val",   "real"): 1000,
    ("val",   "fake"): 1000,
}
# ===================================

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "data" / "raw" / "deepfake-and-real-images"
OUT  = ROOT / "data" / "subsets" / SUBSET_NAME
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def list_images(folder: Path):
    files = []
    for e in EXTS:
        files += folder.glob(e)
    return files

def sample_copy(src: Path, dst: Path, n: int):
    dst.mkdir(parents=True, exist_ok=True)
    pool = list_images(src)
    if not pool:
        raise SystemExit(f"No images found in: {src}")
    n = min(n, len(pool))
    chosen = random.sample(pool, n)
    for p in chosen:
        shutil.copy2(p, dst / p.name)
    return n, len(pool)

def main():
    random.seed(RANDOM_SEED)

    if not SRC.exists():
        raise SystemExit(f"Source dataset not found: {SRC}")

    if OVERWRITE and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # sanity check
    for (split, cls) in COUNTS.keys():
        path = SRC / split / cls
        if not path.exists():
            raise SystemExit(f"Missing folder: {path}")

    # build subset
    print(f"Creating subset at: {OUT}")
    summary = defaultdict(int)
    total = 0
    for (split, cls), n in COUNTS.items():
        src = SRC / split / cls
        dst = OUT / split / cls
        done, avail = sample_copy(src, dst, n)
        summary[(split, cls)] = done
        total += done
        print(f"  {split}/{cls}: {done} files  (requested {n}, available {avail})")

    # final report
    print("\n✅ Subset ready")
    for (split, cls), cnt in sorted(summary.items()):
        print(f"  {split}/{cls}: {cnt}")
    print(f"Total: {total} images")
    print(f"Path : {OUT}")

if __name__ == "__main__":
    main()
