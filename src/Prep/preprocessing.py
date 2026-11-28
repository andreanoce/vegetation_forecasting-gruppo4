import numpy as np

from src.config import (
    CLIMATE_TO_IDX,
    COV_COLS,
    HIST_TARGET_COLS,
    WEATHER_COLS,
)
from src.Prep.scaler import ArcsinhScaler

# Numero di canali finali prodotti dal preprocessing
HISTORY_SEQ_DIM = 24       # 21 feature + is_clear_sky + days_since_clear_sky + weather_missing_frac
FUTURE_SEQ_DIM = 21        # 18 covariate + is_target_day + days_from_now + weather_missing_frac
HIST_STEP_DIM = 22         # 21 feature nei 3 giorni sereni + delta_t
FUTURE_STEP_DIM = 27       # 18 covariate nel giorno target + 8 medie meteo nell'intervallo + delta_t


def _forward_fill_2d(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill vettorizzato lungo l'asse temporale (axis=0) su un array (T, C):
    ogni NaN prende l'ultimo valore osservato prima di lui nella stessa colonna.
    Eventuali NaN prima della prima osservazione vengono posti a 0.0 (media del train).
    """
    n_days = arr.shape[0]
    observed = ~np.isnan(arr)
    last_obs_idx = np.where(observed, np.arange(n_days)[:, None], 0)
    last_obs_idx = np.maximum.accumulate(last_obs_idx, axis=0)
    filled = arr[last_obs_idx, np.arange(arr.shape[1])]
    return np.nan_to_num(filled, nan=0.0).astype(np.float32)


def _safe_nanmean_axis0(arr: np.ndarray) -> np.ndarray:
    """
    Calcola la media lungo axis=0 ignorando i NaN senza emettere RuntimeWarning
    se un'intera colonna è tutta NaN (nel qual caso restituisce 0.0 = media del train).
    """
    valid_mask = ~np.isnan(arr)
    counts = valid_mask.sum(axis=0)
    sums = np.where(valid_mask, arr, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    return means.astype(np.float32)


def preprocess_window(
    history: np.ndarray,
    future: np.ndarray,
    target: np.ndarray,
    future_target_positions: np.ndarray,
    climate: str,
    scaler: ArcsinhScaler,
    target_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Trasforma una singola finestra grezza .npz in tensori numerici puliti (zero NaN).
    """
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)

    # Maschera di validità dei target (utile se l'Hidden Test Set avesse target mascherati)
    if target_mask is not None:
        target_valid = (~np.asarray(target_mask, dtype=bool)) & (~np.isnan(target_arr))
    else:
        target_valid = ~np.isnan(target_arr)
    target_raw = np.nan_to_num(target_arr, nan=0.0).astype(np.float32)

    ftp = np.asarray(future_target_positions, dtype=np.int64)

    t_hist = history.shape[0]
    t_fut = future.shape[0]

    # 1. Individuazione dei 3 giorni clear-sky nello storico (dove tutti e 3 gli indici sono validi)
    is_clear_sky_bool = ~np.isnan(history[:, HIST_TARGET_COLS]).any(axis=1)
    is_clear_sky = is_clear_sky_bool.astype(np.float32)
    clear_sky_indices = np.flatnonzero(is_clear_sky_bool)

    if len(clear_sky_indices) == 0:
        clear_sky_indices = np.array([0, t_hist // 2, t_hist - 1], dtype=np.int64)
    elif len(clear_sky_indices) < 3:
        clear_sky_indices = np.pad(clear_sky_indices, (0, 3 - len(clear_sky_indices)), mode="edge")
    elif len(clear_sky_indices) > 3:
        clear_sky_indices = clear_sky_indices[-3:]

    # Ultimo valore clear-sky osservato al tempo t (scala originale, shape (3,))
    last_clear_target_raw = np.nan_to_num(
        history[clear_sky_indices[-1], HIST_TARGET_COLS].copy(),
        nan=0.0,
    )

    # 2. Normalizzazione con lo scaler del train
    hist_norm = scaler.transform_features(history)                           # (T_hist, 21)
    fut_cov_norm = scaler.transform_features(future[:, COV_COLS], COV_COLS)  # (T_fut, 18)
    target_norm = np.nan_to_num(scaler.transform_target(target_raw), nan=0.0)  # (3, 3)
    # Normalizzato con transform_target (target_mean/target_var) per coerenza esatta con target_norm
    last_clear_target_norm = scaler.transform_target(last_clear_target_raw)  # (3,)

    # 3. Gestione NaN nelle 18 covariate (meteo + nubi + Fourier)
    hist_weather_missing = np.isnan(hist_norm[:, WEATHER_COLS]).mean(axis=1, keepdims=True).astype(np.float32)
    fut_weather_missing = np.isnan(fut_cov_norm[:, WEATHER_COLS]).mean(axis=1, keepdims=True).astype(np.float32)

    hist_cov_clean = np.nan_to_num(hist_norm[:, COV_COLS], nan=0.0)
    fut_cov_clean = np.nan_to_num(fut_cov_norm, nan=0.0)

    # 4. Forward-fill vettorizzato dei 3 indici di vegetazione storici nei giorni nuvolosi
    hist_targets_ffill = _forward_fill_2d(hist_norm[:, HIST_TARGET_COLS])

    # Giorni trascorsi dall'ultima osservazione satellitare serena (normalizzati / 10)
    days_since_clear = np.zeros((t_hist, 1), dtype=np.float32)
    counter = 0.0
    for t in range(t_hist):
        if is_clear_sky_bool[t]:
            counter = 0.0
        else:
            counter += 1.0
        days_since_clear[t, 0] = counter / 10.0

    # Sequenza storica completa: shape (T_hist, 24)
    history_seq = np.concatenate(
        [
            hist_cov_clean,                 # 18 covariate
            hist_targets_ffill,             # 3 indici storici (con forward-fill)
            is_clear_sky[:, None],          # 1 maschera osservazione satellitare
            days_since_clear,               # 1 distanza temporale dall'ultima foto
            hist_weather_missing,           # 1 frazione sensori meteo mancanti
        ],
        axis=1,
    ).astype(np.float32)

    # 5. Costruzione sequenza futura completa: shape (T_fut, 21)
    ftp_clipped = np.clip(ftp, 0, t_fut - 1)
    is_target_day = np.zeros((t_fut, 1), dtype=np.float32)
    is_target_day[ftp_clipped, 0] = 1.0
    trailing_hist_days = float((t_hist - 1) - clear_sky_indices[-1])
    days_from_now = ((np.arange(1, t_fut + 1, dtype=np.float32) + trailing_hist_days)[:, None]) / 15.0

    future_seq = np.concatenate(
        [
            fut_cov_clean,                  # 18 covariate future
            is_target_day,                  # 1 flag giorno di passaggio satellitare
            days_from_now,                  # 1 orizzonte temporale dall'ultimo clear-sky
            fut_weather_missing,            # 1 frazione sensori meteo mancanti
        ],
        axis=1,
    ).astype(np.float32)

    # 6. Rappresentazione compatta a dimensione fissa sui 3 step storici: shape (3, 22)
    hist_step_feats = np.concatenate(
        [hist_cov_clean[clear_sky_indices], hist_targets_ffill[clear_sky_indices]],
        axis=1,
    )
    hist_deltas = np.zeros((3, 1), dtype=np.float32)
    hist_deltas[1, 0] = (clear_sky_indices[1] - clear_sky_indices[0]) / 10.0
    hist_deltas[2, 0] = (clear_sky_indices[2] - clear_sky_indices[1]) / 10.0
    hist_clear_steps = np.concatenate([hist_step_feats, hist_deltas], axis=1).astype(np.float32)

    # 7. Rappresentazione compatta sui 3 intervalli futuri: shape (3, 27)
    future_step_list = []
    prev_idx = 0
    for step_i, pos in enumerate(ftp_clipped):
        pos_int = int(pos)
        raw_interval_weather = fut_cov_norm[prev_idx : pos_int + 1, WEATHER_COLS]
        weather_interval_mean = _safe_nanmean_axis0(raw_interval_weather)   # (8,)
        cov_at_target = fut_cov_clean[pos_int]                              # (18,)
        extra_gap = trailing_hist_days if step_i == 0 else 0.0
        delta_days = np.array([(pos_int - prev_idx + 1 + extra_gap) / 10.0], dtype=np.float32)  # (1,)
        future_step_list.append(np.concatenate([cov_at_target, weather_interval_mean, delta_days]))
        prev_idx = pos_int + 1

    future_step_features = np.stack(future_step_list, axis=0).astype(np.float32)

    # 8. Codifica della classe climatica di Köppen (con fallback su 'unknown' = 0)
    climate_idx_val = CLIMATE_TO_IDX.get(str(climate), 0)
    climate_idx = np.int64(climate_idx_val)

    return {
        "history_seq": history_seq,                         # (T_hist, 24)
        "future_seq": future_seq,                           # (T_fut, 21)
        "hist_clear_steps": hist_clear_steps,               # (3, 22)
        "future_step_features": future_step_features,       # (3, 27)
        "future_target_positions": ftp_clipped,             # (3,)
        "last_clear_target_norm": last_clear_target_norm,   # (3,)
        "last_clear_target_raw": last_clear_target_raw,     # (3,)
        "climate_idx": climate_idx,                         # scalare int64
        "target_norm": target_norm,                         # (3, 3)
        "target_raw": target_raw,                           # (3, 3)
        "target_valid": target_valid.astype(np.float32),    # (3, 3) 1.0 = valido, 0.0 = mancante
    }