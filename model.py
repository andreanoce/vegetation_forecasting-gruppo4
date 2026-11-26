"""
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
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import VI_HISTORY_COLS, Preprocessor

QUANTILES = (0.1, 0.5, 0.9)
N_VI = 3


def meteo_columns(pre: Preprocessor) -> list[int]:
    """Le 18 feature originali di meteo + giorno dell'anno (tutte tranne gli indici).
    Le feature ingegnerizzate non entrano: il GRU-D calcola da solo ffill e giorni
    dall'ultima osservazione."""
    original = [pre.feature_names[i] for i in range(len(pre.keep_idx))]
    return [pre.feature_names.index(n) for n in original if n not in VI_HISTORY_COLS]


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
    def __init__(self, pre: Preprocessor, channels=64, kernel=5, hidden=128,
                 dropout=0.1, n_steps=3):
        super().__init__()
        self.meteo_idx = meteo_columns(pre)
        self.vi_idx = [pre.feature_names.index(c) for c in VI_HISTORY_COLS]

        self.cnn = MeteoCNN(len(self.meteo_idx), channels, kernel, dropout)
        self.encoder = GRUDEncoder(channels, N_VI, hidden)
        self.decoder = nn.GRU(channels, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # una testa per passo: 3 mediane (delta) + 3 distanze basse + 3 distanze alte
        self.heads = nn.ModuleList(nn.Linear(hidden + N_VI, 3 * N_VI) for _ in range(n_steps))

        # gli indici nella history sono normalizzati con lo scaler delle FEATURE, i target
        # con quello dei TARGET: servono entrambi per esprimere l'ultimo valore osservato
        # nella scala dei target (la mediana e' una variazione rispetto a esso).
        fs = pre.feat_scaler.subset(self.vi_idx)
        ts = pre.target_scaler
        for name, v in [("f_mean", fs.mean), ("f_std", fs.std),
                        ("t_mean", ts.mean), ("t_std", ts.std)]:
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
        vi_m = b["history_valid"][..., self.vi_idx]

        h_emb, f_emb = self.cnn(h_met), self.cnn(f_met)             # 1. CNN condivisa
        h_enc, x_last = self.encoder(h_emb, vi, vi_m, b["history_pad"])  # 2. GRU-D
        dec, _ = self.decoder(f_emb, h_enc.unsqueeze(0).contiguous())    # 3. decoder

        idx = b["target_pos"].unsqueeze(-1).expand(-1, -1, dec.size(-1))
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
