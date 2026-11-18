import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from bytenet.dataset import FragmentDataset, mixup
from bytenet.models import ByteNet

RATES = [0.0, 0.05, 0.10, 0.20, 0.30]

def evaluate(model, ds, abl, bs=128):
    model.eval(); correct = 0
    with torch.no_grad():
        for raw, img, st, lab in DataLoader(ds, batch_size=bs):
            if abl == "image_only": raw, st = torch.zeros_like(raw), torch.zeros_like(st)
            correct += (model(raw, img, st).argmax(1) == lab).sum().item()
    return correct / len(ds)

def corrupt(X, rate, seed=123):
    rng = np.random.default_rng(seed)
    m = rng.random(X.shape) < rate
    return np.where(m, rng.integers(0, 256, X.shape), X).astype(np.uint8)

def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load(a.data, allow_pickle=True); X, y = d["X"], d["y"]; names = list(d["class_names"]); ns = int(d["sector_size"])
    idx = np.random.default_rng(a.seed).permutation(len(X)); nt = int(len(X) * 0.2)
    te, tr = idx[:nt], idx[nt:]
    tr_ds = FragmentDataset(X[tr], y[tr], a.n_gram, True); te_ds = FragmentDataset(X[te], y[te], a.n_gram, False)
    loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    model = ByteNet(ns, ns - a.n_gram + 1, 8 * a.n_gram, len(names), a.variant, a.side, a.fusion,
                    skip_image=(a.ablation == "side_only"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    wu = max(1, a.epochs // 6)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/wu if e < wu else 0.5*(1+np.cos(np.pi*(e-wu)/max(1, a.epochs-wu))))
    hist = {"train_loss": [], "test_acc": []}
    for ep in range(a.epochs):
        t0 = time.time(); model.train(); ls = n = 0
        for raw, img, st, lab in loader:
            if a.ablation == "image_only": raw, st = torch.zeros_like(raw), torch.zeros_like(st)
            raw, img, st, ysoft = mixup(raw, img, st, lab, len(names))
            opt.zero_grad()
            loss = -(ysoft * F.log_softmax(model(raw, img, st), 1)).sum(1).mean()
            loss.backward(); opt.step(); ls += loss.item() * raw.size(0); n += raw.size(0)
        sch.step(); acc = evaluate(model, te_ds, a.ablation)
        hist["train_loss"].append(ls / n); hist["test_acc"].append(acc)
        print(f"[{a.name}] epoch {ep+1}/{a.epochs} loss={ls/n:.4f} test_acc={acc:.3f} ({time.time()-t0:.1f}s)", flush=True)
    rob = {}
    for r in RATES:
        ds = te_ds if r == 0 else FragmentDataset(corrupt(X[te], r), y[te], a.n_gram, False)
        rob[str(r)] = evaluate(model, ds, a.ablation)
    # confusion matrix
    model.eval(); conf = np.zeros((len(names),) * 2, dtype=int)
    with torch.no_grad():
        for raw, img, st, lab in DataLoader(te_ds, batch_size=128):
            if a.ablation == "image_only": raw, st = torch.zeros_like(raw), torch.zeros_like(st)
            for t, p in zip(lab.tolist(), model(raw, img, st).argmax(1).tolist()): conf[t, p] += 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"name": a.name, "args": vars(a), "class_names": names, "history": hist, "final_test_acc": hist["test_acc"][-1],
               "robustness": rob, "confusion_matrix": conf.tolist(), "n_params": sum(p.numel() for p in model.parameters()),
               "n_train": len(tr), "n_test": len(te)}, open(out / "result.json", "w"), indent=1)
    print(f"[{a.name}] FINAL acc={hist['test_acc'][-1]:.4f} robustness={rob}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fragments.npz"); ap.add_argument("--name", default="run")
    ap.add_argument("--variant", default="resnet", choices=["resnet", "poolformer"])
    ap.add_argument("--side", default="byte", choices=["byte", "hist"])
    ap.add_argument("--fusion", default="concat", choices=["concat", "gate"])
    ap.add_argument("--ablation", default="full", choices=["full", "side_only", "image_only"])
    ap.add_argument("--n-gram", type=int, default=4); ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/run")
    run(ap.parse_args())
