"""Builds a real 7-class fragment corpus by carving sectors from random offsets inside local files."""
import argparse, random
from pathlib import Path
import numpy as np

GLOBS = {"py": ["*.py"], "json": ["*.json"], "html": ["*.html"], "png": ["*.png"],
         "pdf": ["*.pdf"], "gz": ["*.gz"], "elf": None}
ROOTS = ["/usr", "/opt", "/home", "/etc", "/root"]

def iter_files(globs, min_size):
    seen = set()
    for r in ROOTS:
        rp = Path(r)
        if not rp.exists(): continue
        for pat in globs:
            for p in rp.rglob(pat):
                if p in seen or not p.is_file() or p.is_symlink(): continue
                try:
                    if p.stat().st_size >= min_size: seen.add(p); yield p
                except OSError: pass

def iter_elf(min_size):
    seen = set()
    for r in ["/usr/bin", "/usr/lib", "/usr/local/bin", "/usr/local/lib"]:
        rp = Path(r)
        if not rp.exists(): continue
        for p in rp.rglob("*"):
            try:
                if not p.is_file() or p.is_symlink() or p in seen or p.stat().st_size < min_size: continue
                with open(p, "rb") as f:
                    if f.read(4) == b"\x7fELF": seen.add(p); yield p
            except OSError: pass

def build(sector, per_class, seed):
    rng = np.random.default_rng(seed); random.seed(seed)
    names = sorted(GLOBS); X, y, rep = [], [], {}
    for lab, c in enumerate(names):
        files = list(iter_elf(2*sector) if c == "elf" else iter_files(GLOBS[c], 2*sector))
        random.shuffle(files); files = files[:4000]
        secs, tries = [], 0
        while files and len(secs) < per_class and tries < per_class*20:
            tries += 1
            f = files[rng.integers(0, len(files))]
            try:
                sz = f.stat().st_size
                off = int(rng.integers(0, sz - sector + 1))
                with open(f, "rb") as fh: fh.seek(off); ch = fh.read(sector)
                if len(ch) == sector: secs.append(np.frombuffer(ch, dtype=np.uint8))
            except OSError: pass
        rep[c] = (len(secs), len(files))
        X += secs; y += [lab]*len(secs)
    return np.stack(X), np.array(y), names, rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector-size", type=int, default=256)
    ap.add_argument("--per-class", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/fragments.npz")
    a = ap.parse_args()
    X, y, names, rep = build(a.sector_size, a.per_class, a.seed)
    np.savez_compressed(a.out, X=X, y=y, class_names=np.array(names), sector_size=a.sector_size)
    print(f"Saved {len(X)} sectors of size {a.sector_size} bytes -> {a.out}")
    print("Classes:", names)
    for c, (n, nf) in rep.items(): print(f"  {c:5s}: {n:4d} sectors from {nf:4d} files")
