"""
Valutazione di un checkpoint come richiesto dalla traccia, per ogni indice e ogni split
di valutazione (id, ood-s, ood-t), in scala originale:

  Point accuracy (mediana q = 0.5) : RMSE, MAE, WMAPE, MASE
  Uncertainty (quantili 0.1/0.5/0.9): CRPS, pinball, coverage

Dati preparati da preprocess.py con lo stesso scaler del training (scaler.json del train).

  python evaluate.py                         # usa checkpoints/cnn_grud_best.pt
  python evaluate.py --ckpt altro.pt --out risultati.json
  python evaluate.py --model cnn_grud --ckpt ...  # se il checkpoint non salva il nome
"""

from __future__ import annotations

import argparse

import torch

from metrics import format_table, metrics_from_model, save_json
from preprocess import build_datasets, make_loader
from main import MODELS, get_device, load_model_module

SPLITS = [("id", "id"), ("ood-s", "ood_s"), ("ood-t", "ood_t")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: il checkpoint del modello")
    ap.add_argument("--model", default=None, choices=list(MODELS),
                    help="default: il nome salvato nel checkpoint, altrimenti cnn_grud")
    ap.add_argument("--out", default=None, help="file .json dove salvare le metriche")
    args = ap.parse_args()
    device = get_device()

    data = build_datasets(include_official_splits=True)
    scaler, feature_names = data["scaler"], data["train"].feature_names

    if args.ckpt is None:
        args.ckpt = str(load_model_module(args.model or "cnn_grud").CKPT)
    ck = torch.load(args.ckpt, map_location=device)
    module = load_model_module(args.model or ck.get("name", "cnn_grud"))
    model = module.build_model(feature_names, scaler, ck.get("args", {})).to(device)   # iperparametri dal checkpoint
    model.load_state_dict(ck["model"])
    print(f"Checkpoint: {args.ckpt} (epoca {ck.get('epoch', '?')})\n")

    results = {}
    for name, key in SPLITS:
        r = module.run_epoch(model, make_loader(data[key], 1024), device)
        results[name] = metrics_from_model(r["q"], r["y"], r["valid"], r["persist"], scaler)
        print(format_table(results[name], f"== {name} ({len(data[key])} finestre) ==") + "\n")

    if args.out:
        save_json(results, args.out)
        print(f"Metriche salvate in {args.out}")


if __name__ == "__main__":
    main()
