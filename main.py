"""
Pipeline: dati -> training -> valutazione -> salvataggio metriche.

  python main.py                                  # cnn_grud, training + val interna
  python main.py --model cnn_grud --eval-official # alla fine anche id / ood-s / ood-t
  python main.py --epochs 1 --limit 2000          # prova veloce
  python main.py --model cnn_grud --help          # iperparametri del modello

Ogni modello sta in un file con il suo nome (modello + training) e va registrato
in MODELS. Il file deve esporre: NAME, CKPT, build_parser(ap), build_model(),
fit(args, data, device), evaluate(model, loader, device, scaler), run_epoch().
"""

from __future__ import annotations

import argparse
import importlib
import time

import torch

from metrics import format_table, save_json
from preprocess import build_datasets, make_loader

MODELS = {
    "cnn_grud": "cnn_grud",       # nome da riga di comando -> file .py
}
OFFICIAL_SPLITS = [("id", "id"), ("ood-s", "ood_s"), ("ood-t", "ood_t")]


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.version.hip:
        # ROCm su Windows (GPU AMD): MIOpen non riesce a compilare i suoi kernel a runtime
        # (manca <type_traits> per hiprtc) -> si usano i kernel nativi di PyTorch.
        torch.backends.cudnn.enabled = False
    return device


def load_model_module(name: str):
    if name not in MODELS:
        raise SystemExit(f"Modello sconosciuto '{name}'. Disponibili: {', '.join(MODELS)}")
    return importlib.import_module(MODELS[name])


def parse_args():
    # 1° passaggio: solo --model, per sapere quali iperparametri aggiungere
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="cnn_grud", choices=list(MODELS))
    name = pre.parse_known_args()[0].model
    module = load_model_module(name)

    # 2° passaggio: opzioni della pipeline + iperparametri del modello scelto
    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-official", action="store_true",
                    help="valuta anche sugli split ufficiali id / ood-s / ood-t")
    module.build_parser(ap)
    return ap.parse_args(), module


def main():
    args, module = parse_args()
    device = get_device()
    print(f"Modello: {module.NAME}  device: {device}")

    # 1. Dati
    t0 = time.time()
    data = build_datasets(include_official_splits=args.eval_official)
    info = data["split_info"]
    print(f"Dati pronti in {time.time() - t0:.0f}s: train {info['n_train']} "
          f"(fino al {info['train_period'][1]}), val {info['n_val']} "
          f"(dal {info['val_period'][0]})")

    # 2. Training (early stopping sulla validazione interna)
    model, va, best_ep = module.fit(args, data, device)
    print(f"\nMiglior modello (epoca {best_ep}, val loss {va['loss']:.4f}) salvato in {args.ckpt}")

    # 3. Valutazione in scala originale
    results = {"val_interna": va}
    if args.eval_official:
        for name, key in OFFICIAL_SPLITS:
            results[name] = module.evaluate(model, make_loader(data[key], 1024), device,
                                            data["scaler"])
    print("Metriche in scala originale\n")
    for name, res in results.items():
        print(format_table(res, f"== {name} ==") + "\n")

    # 4. Salvataggio
    metrics_path = args.ckpt.with_suffix(".metrics.json")
    save_json(results, metrics_path)
    print(f"Metriche salvate in {metrics_path}")


if __name__ == "__main__":
    main()
