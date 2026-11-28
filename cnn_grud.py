"""
CNN + GRU-D: modello e training nello stesso file.

Modello ibrido CNN + GRU-D encoder / GRU decoder con uscita a quantili.

  1. CNN causale condivisa sul meteo (history e future)  -> embedding giornaliero (64)
  2. Encoder GRU-D sulla history: embedding meteo + 3 indici + maschera + giorni
     dall'ultima osservazione, con decadimento sia dell'input sia dello stato.
  3. Decoder GRU sul future, inizializzato con lo stato finale dell'encoder.
  4. Stato del decoder nei 3 giorni target + ultimo valore osservato degli indici.
  5. Una testa lineare per passo: mediana come variazione dall'ultimo valore,
     q0.1 / q0.9 come mediana -/+ softplus(.) -> i quantili non si incrociano.

Uscita: (B, 3 passi, 3 indici, 3 quantili) in scala NORMALIZZATA dei target,
con i quantili nell'ordine QUANTILES = (0.1, 0.5, 0.9).
Ingresso: i batch di preprocess.collate_batch (21 feature originali + maschere).

Training: pinball media sui quantili 0.1/0.5/0.9, solo sui target validi.
AdamW, gradient clipping a 1, ReduceLROnPlateau; early stopping e scheduler guardano
la loss sulla validazione interna (mai id/ood).

Si lancia da main.py:
  python main.py --model cnn_grud
  python main.py --model cnn_grud --epochs 1 --limit 2000   # prova veloce

Interfaccia usata da main.py / evaluate.py (ogni file-modello espone la stessa):
  NAME, CKPT, build_parser(), build_model(), fit(), evaluate(), run_epoch()
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from metrics import QUANTILES, format_short, metrics_from_model
from preprocess import VEGETATION_INDEX_COLUMNS, ArcsinhScaler, make_loader

NAME = "cnn_grud"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "cnn_grud_best.pt"
SEED = 42
N_VI = 3


def meteo_columns(feature_names: list[str]) -> list[int]:
    """Le 18 covariate (meteo, nubi, giorno dell'anno): tutte le colonne tranne i 3 indici."""
    return [i for i, name in enumerate(feature_names) if name not in VEGETATION_INDEX_COLUMNS]


# --------------------------------------------------------------------------- #
# 1. CNN sul meteo
# --------------------------------------------------------------------------- #
class MeteoCNN(nn.Module):
    """Due Conv1d CAUSALI (padding solo a sinistra): il giorno t vede t-k+1..t.
    Cosi' il padding a destra dei batch non contamina gli ultimi giorni reali."""

    def __init__(self, n_in: int, channels: int = 64, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.k = kernel
        self.conv1 = nn.Conv1d(n_in, channels, kernel)
        self.conv2 = nn.Conv1d(channels, channels, kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, F) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.drop(F.relu(self.conv1(F.pad(x, (self.k - 1, 0)))))
        x = self.drop(F.relu(self.conv2(F.pad(x, (self.k - 1, 0)))))
        return x.transpose(1, 2)


# --------------------------------------------------------------------------- #
# 2. Encoder GRU-D (Che et al., 2018)
# --------------------------------------------------------------------------- #
class GRUDEncoder(nn.Module):
    def __init__(self, n_emb: int, n_vi: int = N_VI, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        # decadimento dell'input: diagonale, un tasso per indice
        self.gamma_x_w = nn.Parameter(torch.zeros(n_vi))
        self.gamma_x_b = nn.Parameter(torch.zeros(n_vi))
        # decadimento dello stato nascosto
        self.gamma_h = nn.Linear(n_vi, hidden)
        # input della cella: x_hat, maschera, log(1+delta), embedding meteo
        self.cell = nn.GRUCell(3 * n_vi + n_emb, hidden)
        self.register_buffer("x_mean", torch.zeros(n_vi))   # media empirica (scala normalizzata)

    def forward(self, emb, x, m, pad):
        """emb (B,T,E) embedding meteo; x (B,T,V) indici normalizzati (0 dove mancano);
        m (B,T,V) maschera 1=osservato; pad (B,T) True=padding.
        Ritorna stato finale (B,H) e ultimo valore osservato (B,V) in scala normalizzata."""
        B, T, V = x.shape
        h = emb.new_zeros(B, self.hidden)
        x_last = self.x_mean.expand(B, V).clone()
        delta = emb.new_zeros(B, V)          # giorni dall'ultima osservazione
        m_prev = emb.new_ones(B, V)          # cosi' delta al primo passo vale 0

        for t in range(T):
            delta = torch.where(m_prev > 0, torch.zeros_like(delta), delta) + (t > 0)
            mt, xt = m[:, t], x[:, t]

            gx = torch.exp(-F.relu(self.gamma_x_w * delta + self.gamma_x_b))
            x_hat = mt * xt + (1 - mt) * (gx * x_last + (1 - gx) * self.x_mean)

            gh = torch.exp(-F.relu(self.gamma_h(delta)))
            inp = torch.cat([x_hat, mt, torch.log1p(delta), emb[:, t]], dim=-1)
            h_new = self.cell(inp, gh * h)

            keep = pad[:, t, None]           # sui passi di padding lo stato resta fermo
            h = torch.where(keep, h, h_new)
            x_last = torch.where(mt > 0, xt, x_last)
            m_prev = mt
        return h, x_last


# --------------------------------------------------------------------------- #
# Modello completo
# --------------------------------------------------------------------------- #
class CNNGRUD(nn.Module):
    def __init__(self, feature_names: list[str], scaler: ArcsinhScaler, channels=64,
                 kernel=5, hidden=128, dropout=0.1, n_steps=3):
        super().__init__()
        self.meteo_idx = meteo_columns(feature_names)
        self.vi_idx = [feature_names.index(c) for c in VEGETATION_INDEX_COLUMNS]

        self.cnn = MeteoCNN(len(self.meteo_idx), channels, kernel, dropout)
        self.encoder = GRUDEncoder(channels, N_VI, hidden)
        self.decoder = nn.GRU(channels, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # una testa per passo: 3 mediane (delta) + 3 distanze basse + 3 distanze alte
        self.heads = nn.ModuleList(nn.Linear(hidden + N_VI, 3 * N_VI) for _ in range(n_steps))

        # gli indici nella history sono normalizzati con lo scaler delle FEATURE, i target
        # con quello dei TARGET: servono entrambi per esprimere l'ultimo valore osservato
        # nella scala dei target (la mediana e' una variazione rispetto a esso).
        for name, v in [("f_mean", scaler.feature_mean[self.vi_idx]),
                        ("f_std", scaler.feature_std[self.vi_idx]),
                        ("t_mean", scaler.target_mean), ("t_std", scaler.target_std)]:
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    def set_vi_mean(self, x_mean: torch.Tensor):
        self.encoder.x_mean.copy_(x_mean)

    def _feat_to_target_scale(self, x):
        raw = torch.sinh(x) * self.f_std + self.f_mean
        return torch.asinh((raw - self.t_mean) / self.t_std)

    def forward(self, b: dict) -> torch.Tensor:
        h_met = b["history"][..., self.meteo_idx]
        f_met = b["future"][..., self.meteo_idx]
        vi = b["history"][..., self.vi_idx]
        vi_m = b["history_observed"][..., self.vi_idx]   # 1 = indice osservato quel giorno

        h_emb, f_emb = self.cnn(h_met), self.cnn(f_met)             # 1. CNN condivisa
        h_enc, x_last = self.encoder(h_emb, vi, vi_m, b["history_padding"])  # 2. GRU-D
        dec, _ = self.decoder(f_emb, h_enc.unsqueeze(0).contiguous())    # 3. decoder

        idx = b["target_positions"].unsqueeze(-1).expand(-1, -1, dec.size(-1))
        z = self.drop(torch.gather(dec, 1, idx))                     # 4. (B, 3, H)
        last = self._feat_to_target_scale(x_last)                    # (B, 3) scala target
        z = torch.cat([z, last.unsqueeze(1).expand(-1, z.size(1), -1)], dim=-1)

        out = torch.stack([head(z[:, s]) for s, head in enumerate(self.heads)], 1)  # 5.
        d_med, d_lo, d_hi = out.split(N_VI, dim=-1)                  # (B, 3, 3) ciascuno
        med = last.unsqueeze(1) + d_med
        q10 = med - F.softplus(d_lo)
        q90 = med + F.softplus(d_hi)
        return torch.stack([q10, med, q90], dim=-1)                  # (B, 3, 3, 3)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pred (B,3,3,Q), target/valid (B,3,3). Media solo sui target validi."""
    tau = pred.new_tensor(QUANTILES)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(tau * err, (tau - 1) * err)
    w = valid.unsqueeze(-1).expand_as(loss)
    return (loss * w).sum() / w.sum().clamp_min(1)


# ===========================================================================
# TRAINING
# ===========================================================================
def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Iperparametri di questo modello. main.py li aggiunge al proprio parser."""
    ap = ap or argparse.ArgumentParser()
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
    return ap


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
            torch.save(dict(model=model.state_dict(), args=hp, epoch=ep, name=NAME), args.ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                if verbose:
                    print(f"Early stopping: nessun miglioramento da {args.patience} epoche")
                break

    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    return model, evaluate(model, va_loader, device, scaler), best_ep
