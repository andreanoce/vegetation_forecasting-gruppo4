import os
from pathlib import Path

# Percorso principale del progetto (risale automaticamente dalla cartella src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    """
    Risolve il percorso della cartella dati in modo compatibile con Windows, Linux e Colab:
      1. Variabile d'ambiente GREENEARTHNET_DATA_DIR (se impostata)
      2. PROJECT_ROOT / "DATA"
      3. Fallback su PROJECT_ROOT / "Prep" per filesystem case-sensitive (Linux/macOS)
    """
    env_dir = os.getenv("GREENEARTHNET_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    upper_dir = PROJECT_ROOT / "DATA"
    lower_dir = PROJECT_ROOT / "Prep"
    if not upper_dir.exists() and lower_dir.exists():
        return lower_dir
    return upper_dir


DATA_DIR = _resolve_data_dir()

# Sottocartelle degli split ufficiali
TRAIN_DIR = DATA_DIR / "train_multi_output_clear_sky"
VAL_ID_DIR = DATA_DIR / "val_chopped_multi_output_clear_sky"
OOD_S_DIR = DATA_DIR / "ood-s_chopped_multi_output_clear_sky"
OOD_T_DIR = DATA_DIR / "ood-t_chopped_multi_output_clear_sky"

# File dello scaler ufficiale (calcolato SOLO sul train)
TRAIN_SCALER_PATH = TRAIN_DIR / "scaler.json"

# Seed globale per riproducibilità
SEED = 42

# Orizzonte e quantili richiesti dalla traccia
HISTORY_CLEAR_SKY_STEPS = 3
FORECAST_CLEAR_SKY_STEPS = 3
QUANTILES = (0.1, 0.5, 0.9)

# Le 21 feature in ordine esatto come nei file .npz
FEATURE_NAMES = [
    "wind",                                # 0  - Meteo
    "humidity",                            # 1  - Meteo (può avere NaN)
    "solar_irradiance",                    # 2  - Meteo (può avere NaN)
    "rainfall",                            # 3  - Meteo (può avere NaN)
    "pressure",                            # 4  - Meteo (può avere NaN)
    "avg_temperature",                     # 5  - Meteo (può avere NaN)
    "min_temperature",                     # 6  - Meteo (può avere NaN)
    "max_temperature",                     # 7  - Meteo (può avere NaN)
    "thick_clouds",                        # 8  - Nubi
    "thin_clouds",                         # 9  - Nubi
    "clouds_shadows",                      # 10 - Nubi
    "day_of_year",                         # 11 - Temporale
    "day_of_year_fourier_sin_1",           # 12 - Temporale
    "day_of_year_fourier_cos_1",           # 13 - Temporale
    "day_of_year_fourier_sin_2",           # 14 - Temporale
    "day_of_year_fourier_cos_2",           # 15 - Temporale
    "day_of_year_fourier_sin_3",           # 16 - Temporale
    "day_of_year_fourier_cos_3",           # 17 - Temporale
    "target_history_avg_NDVI_clear_sky",   # 18 - Storico Indici (solo nei 3 giorni sereni)
    "target_history_avg_GNDVI_clear_sky",  # 19 - Storico Indici (solo nei 3 giorni sereni)
    "target_history_avg_EVI_clear_sky",    # 20 - Storico Indici (solo nei 3 giorni sereni)
]

# Indici di colonna utili nel preprocessing
WEATHER_COLS = list(range(0, 8))        # 0..7   (8 variabili meteo)
CLOUD_COLS = list(range(8, 11))         # 8..10  (3 variabili nubi)
TEMPORAL_COLS = list(range(11, 18))     # 11..17 (7 variabili calendario/Fourier)
COV_COLS = list(range(0, 18))           # 0..17  (18 covariate note nel passato e nel futuro)
HIST_TARGET_COLS = list(range(18, 21))  # 18..20 (3 indici satellitari passati)

# I 3 target da prevedere
TARGET_NAMES = [
    "avg_NDVI_clear_sky",
    "avg_GNDVI_clear_sky",
    "avg_EVI_clear_sky",
]

# Classi climatiche di Köppen presenti nel dataset ("unknown" all'indice 0 per gestire ood-s)
CLIMATE_CLASSES = [
    "unknown",
    "BSk",
    "Cfa",
    "Cfb",
    "Csa",
    "Csb",
    "Dfa",
    "Dfb",
    "Dfc",
]
CLIMATE_TO_IDX = {c: i for i, c in enumerate(CLIMATE_CLASSES)}