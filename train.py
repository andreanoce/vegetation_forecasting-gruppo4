"""
Training del modello CNN + GRU-D (model.py).

  python train.py                       # training completo con early stopping
  python train.py --epochs 1 --limit 2000   # prova veloce
  python train.py --eval-official       # alla fine valuta anche val_id / ood_t / ood_s

Loss: pinball media sui quantili 0.1/0.5/0.9, solo sui target validi.
Ottimizzatore: AdamW (lr 1e-3, wd 1e-4), batch 256, gradient clipping a 1.
Early stopping sulla validazione interna (minicubi tenuti fuori da group_split).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from model import CNNGRUD, pinball_loss
from preprocessing import (SEED, TARGET_SHORT, VI_HISTORY_COLS, build_datasets,
                           compute_metrics, make_loader)

CKPT = Path(__file__).resolve().parent / "checkpoints" / "cnn_grud_best.pt"


def vi_mean(ds, pre) -> torch.Tensor:
    """Media empirica degli indici osservati nel train (scala normalizzata), per GRU-D."""
    cols = [pre.feature_names.index(c) for c in VI_HISTORY_COLS]
    s, n = np.zeros(3), np.zeros(3)
    for smp in ds.samples:
        x, v = smp["history"][:, cols], smp["history_valid"][:, cols]
        s += (x * v).sum(0); n += v.sum(0)
    return torch.tensor(s / np.maximum(n, 1), dtype=torch.float32)


def run_epoch(model, loader, device, opt=None, clip=1.0):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    preds, ys, ms = [], [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            q = model(b)
            loss = pinball_loss(q, b["target"], b["target_valid"])
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            w = b["target_valid"].sum().item()
            tot += loss.item() * w; n += w
            if not train:
                preds.append(q.cpu()); ys.append(b["target_raw"].cpu()); ms.append(b["target_valid"].cpu())
    out = dict(loss=tot / max(n, 1))
    if not train:
        out.update(q=torch.cat(preds), y=torch.cat(ys), m=torch.cat(ms))
    return out


def evaluate(model, loader, device, pre) -> dict:
    r = run_epoch(model, loader, device)
    met = compute_metrics(r["q"][..., 1], r["y"], r["m"], pre)   # metriche sulla mediana
    # copertura dell'intervallo 80% (atteso ~0.80), in scala originale
    lo, hi = pre.inverse_target(r["q"][..., 0].double()), pre.inverse_target(r["q"][..., 2].double())
    y, m = r["y"].double(), r["m"].bool()
    met["coverage80"] = ((y >= lo) & (y <= hi))[m].float().mean().item()
    met["pinball"] = r["loss"]
    return met


def fmt(met: dict) -> str:
    per = "  ".join(f"{n} {met[n]['RMSE']:.4f}/{met[n]['R2']:.3f}" for n in TARGET_SHORT)
    return (f"pinball {met['pinball']:.4f} | RMSE/R2 {per} | mean RMSE {met['mean_RMSE']:.4f}"
            f" | cov80 {met['coverage80']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="usa solo N finestre (debug)")
    ap.add_argument("--eval-official", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    data = build_datasets(with_official_splits=args.eval_official)
    pre, train_ds, val_ds = data["pre"], data["train"], data["val"]
    print(f"Dati pronti in {time.time() - t0:.0f}s: train {len(train_ds)}, val {len(val_ds)}")

    model = CNNGRUD(pre, kernel=args.kernel, dropout=args.dropout).to(device)
    model.set_vi_mean(vi_mean(train_ds, pre).to(device))
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}  device: {device}")

    tr_set, va_set = train_ds, val_ds
    if args.limit:
        tr_set = Subset(train_ds, range(min(args.limit, len(train_ds))))
        va_set = Subset(val_ds, range(min(args.limit // 4 or 1, len(val_ds))))
    tr_loader = make_loader(tr_set, args.batch, shuffle=True)
    va_loader = make_loader(va_set, 1024)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, bad = float("inf"), 0
    CKPT.parent.mkdir(exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr = run_epoch(model, tr_loader, device, opt)
        va = evaluate(model, va_loader, device, pre)
        improved = va["pinball"] < best
        print(f"ep {ep:3d} {time.time() - t:5.0f}s | train {tr['loss']:.4f} | val {fmt(va)}"
              + ("  *" if improved else ""))
        if improved:
            best, bad = va["pinball"], 0
            torch.save(dict(model=model.state_dict(), args=vars(args), epoch=ep), CKPT)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping: nessun miglioramento da {args.patience} epoche")
                break

    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    print(f"\nMiglior modello (val pinball {best:.4f}) salvato in {CKPT}")
    print("val interna :", fmt(evaluate(model, va_loader, device, pre)))
    if args.eval_official:
        for name in ["val_id", "ood_t", "ood_s"]:
            print(f"{name:11s} :", fmt(evaluate(model, make_loader(data[name], 1024), device, pre)))


if __name__ == "__main__":
    main()
