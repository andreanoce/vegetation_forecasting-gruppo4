"""
Preprocessing per il dataset GreenEarthNet (vegetation forecasting).

Pipeline:
  1. Caricamento dei file <area>.npz / <area>.json di uno split.
  2. Feature engineering (opzionale): selezione + nuove feature.
  3. Normalizzazione "standardization_arcsinh" con lo scaler del TRAIN.
  4. Gestione dei NaN: maschera di validita' + sostituzione con 0 DOPO la normalizzazione.
  5. Split train / validation interno, raggruppato per minicubo (niente leakage).
  6. Padding dinamico delle sequenze nel collate_fn del DataLoader.
  7. Inversione della normalizzazione sulle previsioni + metriche in scala originale.

Convenzioni dei tensori prodotti dal DataLoader (B = batch, T = lunghezza paddata):
  history        (B, Th, F)  float  feature normalizzate, NaN -> 0, padding -> 0
  history_valid  (B, Th, F)  float  1 = valore osservato, 0 = NaN originale o padding
  history_pad    (B, Th)     bool   True = timestep di padding (convenzione key_padding_mask di PyTorch)
  history_len    (B,)        long   lunghezza reale della sequenza
  future, future_valid, future_pad, future_len   idem per la parte futura (meteo previsto)
  target_pos     (B, 3)      long   indici, dentro `future`, dei 3 giorni da prevedere
  target         (B, 3, 3)   float  target normalizzati  [orizzonte, (NDVI, GNDVI, EVI)]
  target_valid   (B, 3, 3)   float  1 = target disponibile
  target_raw     (B, 3, 3)   float  target in scala originale (per le metriche)

Uso rapido:
  python preprocessing.py            # costruisce tutto, stampa statistiche e una baseline
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(__file__).resolve().parent / "DATA" / "DATA"
TRAIN_DIR = DATA_ROOT / "train_multi_output_clear_sky"
VAL_DIR = DATA_ROOT / "val_chopped_multi_output_clear_sky"
OOD_T_DIR = DATA_ROOT / "ood-t_chopped_multi_output_clear_sky"
OOD_S_DIR = DATA_ROOT / "ood-s_chopped_multi_output_clear_sky"

VAL_FRACTION = 0.15   # quota di minicubi del train usati come validazione interna
SEED = 42

# Feature originali da scartare (selezione). Esempio: ["day_of_year"] perche'
# ridondante con le sue componenti di Fourier. Lasciata vuota di default.
DROP_FEATURES: list[str] = []

# Se True aggiunge le feature ingegnerizzate (vedi `engineer_features`).
ADD_ENGINEERED_FEATURES = True

VI_HISTORY_COLS = [
    "target_history_avg_NDVI_clear_sky",
    "target_history_avg_GNDVI_clear_sky",
    "target_history_avg_EVI_clear_sky",
]
ENGINEERED_NAMES = [
    "temp_range",            # max_temperature - min_temperature
    "rel_day",               # giorni rispetto al primo giorno futuro (<0 storia, >=0 futuro)
    "ffill_NDVI",            # ultimo NDVI osservato (forward fill)
    "ffill_GNDVI",
    "ffill_EVI",
    "days_since_vi_obs",     # giorni dall'ultima osservazione degli indici
]


# --------------------------------------------------------------------------- #
# 1. Scaler "standardization_arcsinh"
# --------------------------------------------------------------------------- #
class ArcsinhScaler:
    """y = arcsinh((x - mu) / sqrt(var + eps));  x = sinh(y) * sqrt(var + eps) + mu"""

    def __init__(self, mean, var, eps: float = 1e-5):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.sqrt(np.asarray(var, dtype=np.float64) + eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, y):
        if isinstance(y, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=y.dtype, device=y.device)
            std = torch.as_tensor(self.std, dtype=y.dtype, device=y.device)
            return torch.sinh(y) * std + mean
        return (np.sinh(np.asarray(y, dtype=np.float64)) * self.std + self.mean).astype(np.float32)

    def subset(self, idx) -> "ArcsinhScaler":
        s = ArcsinhScaler.__new__(ArcsinhScaler)
        s.mean, s.std = self.mean[idx], self.std[idx]
        return s

    @staticmethod
    def concat(a: "ArcsinhScaler", b: "ArcsinhScaler") -> "ArcsinhScaler":
        s = ArcsinhScaler.__new__(ArcsinhScaler)
        s.mean, s.std = np.concatenate([a.mean, b.mean]), np.concatenate([a.std, b.std])
        return s


def load_train_scalers(train_dir: Path = TRAIN_DIR) -> tuple[ArcsinhScaler, ArcsinhScaler]:
    """Scaler delle feature e dei target, SEMPRE dal train (anche per val/test)."""
    cfg = json.loads((train_dir / "scaler.json").read_text())
    assert cfg["mode"] == "standardization_arcsinh", cfg["mode"]
    eps = cfg.get("eps", 1e-5)
    return (ArcsinhScaler(cfg["mean"], cfg["var"], eps),
            ArcsinhScaler(cfg["target_mean"], cfg["target_var"], eps))


# --------------------------------------------------------------------------- #
# 2. Caricamento dati grezzi
# --------------------------------------------------------------------------- #
@dataclass
class RawSplit:
    feature_names: list[str]
    target_names: list[str]
    history: list[np.ndarray]          # (Th_i, 21) grezzo, con NaN
    future: list[np.ndarray]           # (Tf_i, 21) grezzo, meteo gia' "rumoroso"
    history_ts: list[np.ndarray]       # datetime64[D]
    future_ts: list[np.ndarray]
    target: np.ndarray                 # (N, 3, 3) grezzo
    target_mask: np.ndarray            # (N, 3, 3) True = target MANCANTE
    target_pos: np.ndarray             # (N, 3)
    area: np.ndarray                   # (N,)
    minicube: np.ndarray               # (N,) nome del CSV di origine
    climate: np.ndarray                # (N,)

    def __len__(self):
        return len(self.target)


def load_split(folder: Path) -> RawSplit:
    folder = Path(folder)
    hist, fut, hts, fts = [], [], [], []
    tgt, tmask, tpos, area, cube, clim = [], [], [], [], [], []
    feat_names = target_names = None

    for npz_path in sorted(folder.glob("*.npz")):
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        if feat_names is None:
            feat_names, target_names = meta["feature_names"], meta["target_names"]
        assert meta["feature_names"] == feat_names, f"ordine feature diverso in {npz_path.name}"

        d = np.load(npz_path, allow_pickle=True)
        n = len(d["target"])
        records = meta["metadata_records"]
        assert len(records) == n, f"metadati non allineati in {npz_path.name}"

        hist += [np.asarray(x, dtype=np.float32) for x in d["history"]]
        fut += [np.asarray(x, dtype=np.float32) for x in d["future"]]
        hts += [np.asarray(x).astype("datetime64[D]") for x in d["history_timestamps"]]
        fts += [np.asarray(x).astype("datetime64[D]") for x in d["future_timestamps"]]
        tgt.append(d["target"].astype(np.float32))
        tmask.append(d["target_mask"].astype(bool))
        tpos.append(d["future_target_positions"].astype(np.int64))
        area += [meta["area"]] * n
        cube += [r["csv"] for r in records]
        clim += [str(c) for c in d["climate"]]
        # NB: `future_noise` NON viene usato: e' il rumore aggiunto al meteo futuro
        # per simulare una previsione, usarlo rivelerebbe il meteo reale.

    return RawSplit(feat_names, target_names, hist, fut, hts, fts,
                    np.concatenate(tgt), np.concatenate(tmask), np.concatenate(tpos),
                    np.array(area), np.array(cube), np.array(clim))


# --------------------------------------------------------------------------- #
# 3. Feature engineering (sui valori GREZZI, prima della normalizzazione)
# --------------------------------------------------------------------------- #
def _forward_fill(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward fill lungo il tempo di x (T, C). Ritorna (valori, giorni dall'ultima obs)."""
    T = len(x)
    observed = ~np.isnan(x[:, 0])
    idx = np.where(observed, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    filled = np.where(idx[:, None] >= 0, x[np.clip(idx, 0, None)], np.nan)
    since = np.where(idx >= 0, np.arange(T) - idx, np.nan).astype(np.float32)
    return filled.astype(np.float32), since


def engineer_features(h, f, h_ts, f_ts, names) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna le feature extra (Th, 6) e (Tf, 6) per una finestra."""
    col = {n: i for i, n in enumerate(names)}
    vi_idx = [col[c] for c in VI_HISTORY_COLS]

    # Storia + futuro concatenati: il forward fill "attraversa" il confine, quindi nel
    # futuro le colonne ffill valgono l'ultimo indice osservato (= baseline di persistenza).
    full = np.concatenate([h, f])
    ts = np.concatenate([h_ts, f_ts])

    temp_range = full[:, col["max_temperature"]] - full[:, col["min_temperature"]]
    rel_day = (ts - f_ts[0]).astype(np.float32)
    vi_ffill, since = _forward_fill(full[:, vi_idx])

    extra = np.column_stack([temp_range, rel_day, vi_ffill, since]).astype(np.float32)
    return extra[: len(h)], extra[len(h):]


def fit_extra_scaler(raw: RawSplit, idx: np.ndarray, feat_scaler: ArcsinhScaler) -> ArcsinhScaler:
    """Scaler per le feature nuove. Le ffill degli indici riusano le statistiche ufficiali
    delle colonne storiche corrispondenti; le altre vengono stimate SOLO sul sotto-train."""
    rows = []
    for i in idx:
        eh, ef = engineer_features(raw.history[i], raw.future[i], raw.history_ts[i],
                                   raw.future_ts[i], raw.feature_names)
        rows += [eh, ef]
    allx = np.concatenate(rows).astype(np.float64)
    mean, var = np.nanmean(allx, 0), np.nanvar(allx, 0)

    vi_idx = [raw.feature_names.index(c) for c in VI_HISTORY_COLS]
    official = feat_scaler.subset(vi_idx)
    mean[2:5] = official.mean
    var[2:5] = official.std ** 2 - 1e-5
    return ArcsinhScaler(mean, var)


# --------------------------------------------------------------------------- #
# 4. Normalizzazione + NaN -> 0 + maschera
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessor:
    feat_scaler: ArcsinhScaler     # sulle colonne selezionate (+ extra)
    target_scaler: ArcsinhScaler
    keep_idx: list[int]            # colonne originali tenute
    feature_names: list[str]       # nomi finali, nell'ordine dei tensori
    add_extra: bool

    @classmethod
    def fit(cls, raw_train: RawSplit, train_idx: np.ndarray,
            drop: list[str] = DROP_FEATURES, add_extra: bool = ADD_ENGINEERED_FEATURES):
        feat_scaler, target_scaler = load_train_scalers()
        keep_idx = [i for i, n in enumerate(raw_train.feature_names) if n not in drop]
        names = [raw_train.feature_names[i] for i in keep_idx]
        scaler = feat_scaler.subset(keep_idx)
        if add_extra:
            scaler = ArcsinhScaler.concat(scaler, fit_extra_scaler(raw_train, train_idx, feat_scaler))
            names += ENGINEERED_NAMES
        return cls(scaler, target_scaler, keep_idx, names, add_extra)

    def _seq(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = ~np.isnan(x)                      # maschera PRIMA di riempire
        y = self.feat_scaler.transform(x)         # NaN restano NaN
        y = np.nan_to_num(y, nan=0.0)             # 0 = media dopo la standardizzazione
        return y.astype(np.float32), valid.astype(np.float32)

    def transform_sample(self, raw: RawSplit, i: int) -> dict:
        h = raw.history[i][:, self.keep_idx]
        f = raw.future[i][:, self.keep_idx]
        if self.add_extra:
            eh, ef = engineer_features(raw.history[i], raw.future[i], raw.history_ts[i],
                                       raw.future_ts[i], raw.feature_names)
            h, f = np.concatenate([h, eh], 1), np.concatenate([f, ef], 1)
        h, hv = self._seq(h)
        f, fv = self._seq(f)

        t_raw = raw.target[i]
        t_valid = ~raw.target_mask[i] & ~np.isnan(t_raw)
        t = np.nan_to_num(self.target_scaler.transform(t_raw), nan=0.0)
        return dict(history=h, history_valid=hv, future=f, future_valid=fv,
                    target_pos=raw.target_pos[i], target=t,
                    target_valid=t_valid.astype(np.float32),
                    target_raw=np.nan_to_num(t_raw, nan=0.0))

    def inverse_target(self, y):
        """Riporta previsioni (..., 3) normalizzate alla scala originale."""
        return self.target_scaler.inverse(y)


# --------------------------------------------------------------------------- #
# 5. Split train / validation per minicubo
# --------------------------------------------------------------------------- #
def group_split(raw: RawSplit, val_fraction=VAL_FRACTION, seed=SEED):
    """Le finestre dello stesso minicubo si sovrappongono nel tempo: se finissero in
    split diversi il modello vedrebbe in train quasi gli stessi dati della validazione.
    Dividiamo quindi per minicubo (CSV di origine), stratificando per classe climatica
    cosi' la validazione ha la stessa distribuzione di climi del train."""
    rng = np.random.default_rng(seed)
    cubes, inv = np.unique(raw.minicube, return_inverse=True)
    cube_climate = np.empty(len(cubes), dtype=object)
    cube_climate[inv] = raw.climate

    val_cubes = np.zeros(len(cubes), dtype=bool)
    for c in np.unique(cube_climate):
        ids = np.flatnonzero(cube_climate == c)
        rng.shuffle(ids)
        val_cubes[ids[: int(round(len(ids) * val_fraction))]] = True

    is_val = val_cubes[inv]
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


# --------------------------------------------------------------------------- #
# 6. Dataset + padding dinamico
# --------------------------------------------------------------------------- #
class VegetationDataset(Dataset):
    """Pre-processa tutto una volta sola (i dati stanno comodamente in RAM)."""

    def __init__(self, raw: RawSplit, pre: Preprocessor, indices=None):
        indices = np.arange(len(raw)) if indices is None else indices
        self.samples = [pre.transform_sample(raw, int(i)) for i in indices]
        self.climate = raw.climate[indices]
        self.minicube = raw.minicube[indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _pad(seqs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right padding con zeri fino alla sequenza piu' lunga DEL BATCH."""
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    T, F = int(lens.max()), seqs[0].shape[1]
    out = torch.zeros(len(seqs), T, F, dtype=torch.float32)
    for k, s in enumerate(seqs):
        out[k, : len(s)] = torch.from_numpy(s)
    pad = torch.arange(T)[None, :] >= lens[:, None]       # True = padding
    return out, pad, lens


def collate_fn(batch: list[dict]) -> dict:
    h, h_pad, h_len = _pad([b["history"] for b in batch])
    hv, _, _ = _pad([b["history_valid"] for b in batch])
    f, f_pad, f_len = _pad([b["future"] for b in batch])
    fv, _, _ = _pad([b["future_valid"] for b in batch])
    stack = lambda k, dt=torch.float32: torch.as_tensor(np.stack([b[k] for b in batch]), dtype=dt)
    return dict(history=h, history_valid=hv, history_pad=h_pad, history_len=h_len,
                future=f, future_valid=fv, future_pad=f_pad, future_len=f_len,
                target_pos=stack("target_pos", torch.long), target=stack("target"),
                target_valid=stack("target_valid"), target_raw=stack("target_raw"))


def make_loader(ds: Dataset, batch_size=256, shuffle=False, **kw) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kw)


# --------------------------------------------------------------------------- #
# 7. Metriche in scala originale
# --------------------------------------------------------------------------- #
TARGET_SHORT = ["NDVI", "GNDVI", "EVI"]


def compute_metrics(pred_norm: torch.Tensor, target_raw: torch.Tensor,
                    target_valid: torch.Tensor, pre: Preprocessor) -> dict:
    """pred_norm: (N, 3, 3) previsioni del modello in scala NORMALIZZATA.
    Si inverte la trasformazione e si calcolano RMSE / MAE / R2 per target,
    ignorando i target mancanti."""
    pred = pre.inverse_target(pred_norm.double())
    y, m = target_raw.double(), target_valid.bool()
    out = {}
    for j, name in enumerate(TARGET_SHORT):
        p, t = pred[..., j][m[..., j]], y[..., j][m[..., j]]
        err = p - t
        ss_res, ss_tot = (err ** 2).sum(), ((t - t.mean()) ** 2).sum()
        out[name] = dict(RMSE=err.pow(2).mean().sqrt().item(), MAE=err.abs().mean().item(),
                         R2=(1 - ss_res / ss_tot).item())
    out["mean_RMSE"] = float(np.mean([out[n]["RMSE"] for n in TARGET_SHORT]))
    return out


# --------------------------------------------------------------------------- #
# Costruzione completa
# --------------------------------------------------------------------------- #
def build_datasets(with_official_splits: bool = False):
    raw_train = load_split(TRAIN_DIR)
    tr_idx, va_idx = group_split(raw_train)
    pre = Preprocessor.fit(raw_train, tr_idx)

    out = dict(pre=pre,
               train=VegetationDataset(raw_train, pre, tr_idx),
               val=VegetationDataset(raw_train, pre, va_idx))
    if with_official_splits:  # stesso Preprocessor (= scaler del train) per tutti
        for name, d in [("val_id", VAL_DIR), ("ood_t", OOD_T_DIR), ("ood_s", OOD_S_DIR)]:
            out[name] = VegetationDataset(load_split(d), pre)
    return out


def _persistence_baseline(loader: DataLoader, pre: Preprocessor) -> dict:
    """Baseline: ogni target = ultimo indice osservato nella storia (colonne ffill).
    Serve anche come test end-to-end della pipeline normalizza -> inverti -> metriche."""
    ffill_names = ["ffill_NDVI", "ffill_GNDVI", "ffill_EVI"]
    use_ffill = all(n in pre.feature_names for n in ffill_names)
    # Le colonne ffill hanno le stesse statistiche delle colonne storiche degli indici,
    # quindi in entrambi i casi si torna alla scala originale con lo stesso scaler.
    vi_cols = [pre.feature_names.index(c) for c in (ffill_names if use_ffill else VI_HISTORY_COLS)]
    preds, ys, ms = [], [], []
    for b in loader:
        rows = torch.arange(len(b["history"]))
        if use_ffill:
            # ultimo timestep reale: le colonne ffill contengono gia' l'ultimo valore osservato
            t_last = b["history_len"] - 1
        else:
            # senza feature ingegnerizzate: ultimo timestep in cui l'indice e' stato osservato
            obs = b["history_valid"][:, :, vi_cols[0]] > 0
            t_last = (obs * torch.arange(obs.shape[1])).argmax(dim=1)
        last = b["history"][rows, t_last][:, vi_cols]
        preds.append(last[:, None, :].repeat(1, 3, 1))   # stessa previsione per i 3 orizzonti
        ys.append(b["target_raw"]); ms.append(b["target_valid"])
    # le colonne storiche sono normalizzate con le statistiche delle FEATURE:
    # torno alla scala originale con lo scaler delle feature, poi rinormalizzo con
    # quello dei target per passare da compute_metrics come farebbe un modello.
    p = torch.cat(preds).double()
    p_raw = pre.feat_scaler.subset(vi_cols).inverse(p)
    p_norm = torch.as_tensor(pre.target_scaler.transform(p_raw.numpy()))
    return compute_metrics(p_norm, torch.cat(ys), torch.cat(ms), pre)


if __name__ == "__main__":
    import time

    t0 = time.time()
    data = build_datasets(with_official_splits=False)
    pre = data["pre"]
    print(f"Pre-processing completato in {time.time() - t0:.1f}s")
    print(f"Train: {len(data['train'])} finestre ({len(set(data['train'].minicube))} minicubi)")
    print(f"Val  : {len(data['val'])} finestre ({len(set(data['val'].minicube))} minicubi)")
    assert not set(data["train"].minicube) & set(data["val"].minicube), "leakage tra split!"
    print(f"Feature ({len(pre.feature_names)}): {pre.feature_names}")

    # verifica che la trasformazione inversa sia esatta
    raw_t = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    assert np.allclose(pre.inverse_target(pre.target_scaler.transform(raw_t)), raw_t, atol=1e-6)

    loader = make_loader(data["train"], batch_size=4, shuffle=True)
    b = next(iter(loader))
    print("\nEsempio di batch:")
    for k, v in b.items():
        print(f"  {k:14s} {tuple(v.shape)}  {v.dtype}")
    print("  lunghezze storia:", b["history_len"].tolist(), " futuro:", b["future_len"].tolist())
    assert torch.isfinite(b["history"]).all() and torch.isfinite(b["future"]).all()

    val_loader = make_loader(data["val"], batch_size=1024)
    m = _persistence_baseline(val_loader, pre)
    print("\nBaseline di persistenza sulla validazione (scala originale):")
    for n in TARGET_SHORT:
        print(f"  {n:6s} RMSE={m[n]['RMSE']:.4f}  MAE={m[n]['MAE']:.4f}  R2={m[n]['R2']:.3f}")
