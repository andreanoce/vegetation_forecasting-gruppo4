"""
Verifica veloce del modello: UNA sola metrica, il MASE sulla mediana (q0.5).

  MASE = MAE(modello) / MAE(persistenza)
  persistenza = "il valore futuro e' uguale all'ultimo indice osservato nella storia"

  MASE < 1  -> il modello batte la baseline (piu' basso = meglio)
  MASE > 1  -> il modello fa peggio che ripetere l'ultimo valore

Tutto in scala originale, solo sui target validi (target_mask escluso),
normalizzando sempre con lo scaler del train.

  python evaluate.py                         # usa checkpoints/cnn_grud_best.pt
  python evaluate.py --ckpt altro.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from model import CNNGRUD
from preprocessing import (OOD_S_DIR, OOD_T_DIR, TARGET_SHORT, TRAIN_DIR, VAL_DIR,
                           VI_HISTORY_COLS, Preprocessor, VegetationDataset, group_split,
                           load_split, make_loader)
from train import CKPT

SPLITS = [("id", VAL_DIR), ("ood-t", OOD_T_DIR), ("ood-s", OOD_S_DIR)]


def persistence(raw) -> np.ndarray:
    """(N, 3, 3): per ogni finestra l'ultimo valore osservato degli indici, ripetuto sui 3 passi."""
    cols = [raw.feature_names.index(c) for c in VI_HISTORY_COLS]
    out = np.full((len(raw), 3), np.nan, dtype=np.float64)
    for i, h in enumerate(raw.history):
        vi = h[:, cols]
        obs = np.flatnonzero(~np.isnan(vi).any(1))
        if len(obs):
            out[i] = vi[obs[-1]]
    return np.repeat(out[:, None, :], 3, axis=1)


@torch.no_grad()
def predict_median(model, ds, pre, device) -> np.ndarray:
    """(N, 3, 3) mediana del modello in scala originale."""
    model.eval()
    preds = []
    for b in make_loader(ds, 1024):
        b = {k: v.to(device) for k, v in b.items()}
        preds.append(model(b)[..., 1].cpu())          # quantile 0.5
    return pre.inverse_target(torch.cat(preds).double()).numpy()


def mase(pred, base, y, valid) -> dict:
    out = {}
    for j, name in enumerate(TARGET_SHORT):
        m = valid[..., j] & ~np.isnan(base[..., j])   # stessi punti per modello e baseline
        mae_model = np.abs(pred[..., j][m] - y[..., j][m]).mean()
        mae_base = np.abs(base[..., j][m] - y[..., j][m]).mean()
        out[name] = mae_model / mae_base
    out["mean"] = float(np.mean([out[n] for n in TARGET_SHORT]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stesso preprocessing del training (scaler del train)
    raw_train = load_split(TRAIN_DIR)
    tr_idx, _ = group_split(raw_train)
    pre = Preprocessor.fit(raw_train, tr_idx)

    ck = torch.load(args.ckpt, map_location=device)
    a = ck.get("args", {})
    model = CNNGRUD(pre, kernel=a.get("kernel", 5), dropout=a.get("dropout", 0.1)).to(device)
    model.load_state_dict(ck["model"])
    print(f"Checkpoint: {args.ckpt} (epoca {ck.get('epoch', '?')})\n")

    print(f"{'split':6s} " + " ".join(f"{n:>7s}" for n in TARGET_SHORT) + "    media")
    for name, folder in SPLITS:
        raw = load_split(folder)
        ds = VegetationDataset(raw, pre)
        y = raw.target.astype(np.float64)
        valid = ~raw.target_mask & ~np.isnan(y)
        r = mase(predict_median(model, ds, pre, device), persistence(raw), y, valid)
        print(f"{name:6s} " + " ".join(f"{r[n]:7.3f}" for n in TARGET_SHORT) + f"  {r['mean']:7.3f}")

    print("\nMASE < 1 = meglio della persistenza. Confronta il valore 'media' tra un run e l'altro.")


if __name__ == "__main__":
    main()
