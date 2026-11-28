"""
Training del modello CNN + GRU-D (model.py) sui dati di preprocess.py.

  python train.py                          # training con early stopping
  python train.py --epochs 1 --limit 2000  # prova veloce
  python train.py --eval-official          # alla fine valuta anche id / ood-s / ood-t
  python train.py --lr 5e-4 --dropout 0.2 --hidden 192

Dati: preprocess.build_datasets() -> split cronologico 80/20 del train con gap,
normalizzazione con scaler.json ufficiale, maschere + forward fill.
Loss: pinball media sui quantili 0.1/0.5/0.9, solo sui target validi.
Ottimizzatore: AdamW, gradient clipping a 1, ReduceLROnPlateau.
Early stopping e scheduler guardano la loss sulla validazione interna (mai id/ood).
Alla fine stampa le metriche di metrics.py in scala originale e le salva accanto
al checkpoint (<checkpoint>.metrics.json).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from metrics import format_short, format_table, metrics_from_model, save_json
from model import CNNGRUD, pinball_loss
from preprocess import VEGETATION_INDEX_COLUMNS, build_datasets, make_loader

CKPT = Path(__file__).resolve().parent / "checkpoints" / "cnn_grud_best.pt"
SEED = 42


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4, help="epoche senza miglioramento prima di fermarsi")
    ap.add_argument("--lr-patience", type=int, default=2,
                    help="epoche senza miglioramento prima di ridurre il learning rate")
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128, help="dimensione delle GRU")
    ap.add_argument("--channels", type=int, default=64, help="filtri della CNN sul meteo")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="usa solo N finestre (debug)")
    ap.add_argument("--ckpt", type=Path, default=CKPT, help="dove salvare il modello migliore")
    ap.add_argument("--eval-official", action="store_true")
    return ap


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.version.hip:
        # ROCm su Windows (GPU AMD): MIOpen non riesce a compilare i suoi kernel a runtime
        # (manca <type_traits> per hiprtc) -> si usano i kernel nativi di PyTorch.
        torch.backends.cudnn.enabled = False
    return device


def build_model(feature_names, scaler, hp: dict) -> CNNGRUD:
    """Costruisce il modello dagli iperparametri (args di train o salvati nel checkpoint)."""
    return CNNGRUD(feature_names, scaler, channels=hp.get("channels", 64),
                   kernel=hp.get("kernel", 5), hidden=hp.get("hidden", 128),
                   dropout=hp.get("dropout", 0.1))


def observed_vi_mean(ds) -> torch.Tensor:
    """Media degli indici REALMENTE osservati nel train (scala normalizzata), per GRU-D.
    Si usano solo i giorni con maschera 1: i valori riempiti col forward fill non contano."""
    cols = [ds.feature_names.index(c) for c in VEGETATION_INDEX_COLUMNS]
    total, count = np.zeros(len(cols)), np.zeros(len(cols))
    for s in ds.samples:
        x, observed = s["history"][:, cols], s["history_observed"][:, cols]
        total += (x * observed).sum(0); count += observed.sum(0)
    return torch.tensor(total / np.maximum(count, 1), dtype=torch.float32)


def run_epoch(model, loader, device, opt=None, clip=1.0) -> dict:
    """Un passaggio sul loader. Con `opt` allena, senza valuta e raccoglie le previsioni."""
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    preds, ys, valids, persist = [], [], [], []
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
            total += loss.item() * w; n += w
            if not train:
                preds.append(q.cpu()); ys.append(b["target_raw"].cpu())
                valids.append(b["target_valid"].cpu()); persist.append(b["last_observed_raw"].cpu())
    out = dict(loss=total / max(n, 1))
    if not train:
        out.update(q=torch.cat(preds), y=torch.cat(ys), valid=torch.cat(valids),
                   persist=torch.cat(persist))
    return out


def evaluate(model, loader, device, scaler) -> dict:
    """Metriche in scala originale (metrics.py) + "loss" = pinball in scala normalizzata,
    la stessa ottimizzata in training (usata da early stopping e scheduler)."""
    r = run_epoch(model, loader, device)
    met = metrics_from_model(r["q"], r["y"], r["valid"], r["persist"], scaler)
    met["loss"] = r["loss"]
    return met


def fit(args, data: dict, device, verbose: bool = True):
    """Allena con early stopping sulla validazione interna e salva il migliore in args.ckpt.
    Ritorna (modello migliore, metriche sulla validazione interna, epoca migliore)."""
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    scaler, train_ds, val_ds = data["scaler"], data["train"], data["val"]

    model = build_model(train_ds.feature_names, scaler, vars(args)).to(device)
    model.set_vi_mean(observed_vi_mean(train_ds).to(device))
    if verbose:
        print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}  device: {device}")

    tr_set, va_set = train_ds, val_ds
    if args.limit:
        tr_set = Subset(train_ds, range(min(args.limit, len(train_ds))))
        va_set = Subset(val_ds, range(min(args.limit // 4 or 1, len(val_ds))))
    tr_loader = make_loader(tr_set, args.batch, shuffle=True)   # shuffle solo DOPO lo split
    va_loader = make_loader(va_set, 1024)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    best, bad, best_ep = float("inf"), 0, 0
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr = run_epoch(model, tr_loader, device, opt)
        va = evaluate(model, va_loader, device, scaler)
        sched.step(va["loss"])
        improved = va["loss"] < best
        if verbose:
            print(f"ep {ep:3d} {time.time() - t:5.0f}s | lr {opt.param_groups[0]['lr']:.1e}"
                  f" | train {tr['loss']:.4f} | val {va['loss']:.4f} | {format_short(va)}"
                  + ("  *" if improved else ""), flush=True)
        if improved:
            best, bad, best_ep = va["loss"], 0, ep
            hp = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            torch.save(dict(model=model.state_dict(), args=hp, epoch=ep), args.ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                if verbose:
                    print(f"Early stopping: nessun miglioramento da {args.patience} epoche")
                break

    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    return model, evaluate(model, va_loader, device, scaler), best_ep


def main():
    args = build_parser().parse_args()
    device = get_device()

    t0 = time.time()
    data = build_datasets(include_official_splits=args.eval_official)
    info = data["split_info"]
    print(f"Dati pronti in {time.time() - t0:.0f}s: train {info['n_train']} "
          f"(fino al {info['train_period'][1]}), val {info['n_val']} "
          f"(dal {info['val_period'][0]})")

    model, va, best_ep = fit(args, data, device)
    print(f"\nMiglior modello (epoca {best_ep}, val loss {va['loss']:.4f}) salvato in {args.ckpt}")
    print("Metriche in scala originale\n")
    results = {"val_interna": va}
    if args.eval_official:
        for name, key in [("id", "id"), ("ood-s", "ood_s"), ("ood-t", "ood_t")]:
            results[name] = evaluate(model, make_loader(data[key], 1024), device, data["scaler"])
    for name, res in results.items():
        print(format_table(res, f"== {name} ==") + "\n")
    metrics_path = args.ckpt.with_suffix(".metrics.json")
    save_json(results, metrics_path)
    print(f"Metriche salvate in {metrics_path}")


if __name__ == "__main__":
    main()
