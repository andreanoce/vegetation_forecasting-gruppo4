from pathlib import Path

# Percorso principale del progetto, risale automaticamente dalla cartella src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"

# Sottocartelle degli split.
TRAIN_DIR = DATA_DIR / "train_multi_output_clear_sky"
VAL_ID_DIR = DATA_DIR / "val_chopped_multi_output_clear_sky"
OOD_S_DIR = DATA_DIR / "ood-s_chopped_multi_output_clear_sky"
OOD_T_DIR = DATA_DIR / "ood-t_chopped_multi_output_clear_sky"

# File dello scaler ufficiale calcolato SOLO sul train.
TRAIN_SCALER_PATH = TRAIN_DIR / "scaler.json"

# Seed globale per riproducibilità.
SEED = 42

# Orizzonte e quantili richiesti dalla traccia.
HISTORY_CLEAR_SKY_STEPS = 3
FORECAST_CLEAR_SKY_STEPS = 3
QUANTILES = (0.1, 0.5, 0.9)

# Le 21 feature in ordine esatto come nei file .npz
FEATURE_NAMES = [
    "wind",                               #Meteo
    "humidity",                           #Meteo (può avere NaN)
    "solar_irradiance",                   #Meteo (può avere NaN)
    "rainfall",                           #Meteo (può avere NaN)
    "pressure",                           #Meteo (può avere NaN)
    "avg_temperature",                    #Meteo (può avere NaN)
    "min_temperature",                    #Meteo (può avere NaN)
    "max_temperature",                    #Meteo (può avere NaN)
    "thick_clouds",                       #Nubi
    "thin_clouds",                        #Nubi
    "clouds_shadows",                     #Nubi
    "day_of_year",                        #Temporale
    "day_of_year_fourier_sin_1",          #Temporale
    "day_of_year_fourier_cos_1",          #Temporale
    "day_of_year_fourier_sin_2",          #Temporale
    "day_of_year_fourier_cos_2",          #Temporale
    "day_of_year_fourier_sin_3",          #Temporale
    "day_of_year_fourier_cos_3",          #Temporale
    "target_history_avg_NDVI_clear_sky",  #Storico Indici (presente solo nei 3 giorni sereni)
    "target_history_avg_GNDVI_clear_sky", #Storico Indici (presente solo nei 3 giorni sereni)
    "target_history_avg_EVI_clear_sky",   #Storico Indici (presente solo nei 3 giorni sereni)
]

# Indici di colonna utili nel preprocessing.
WEATHER_COLS = list(range(0, 8))       # 0..7   (8 variabili meteo)
CLOUD_COLS = list(range(8, 11))        # 8..10  (3 variabili nubi)
TEMPORAL_COLS = list(range(11, 18))    # 11..17 (7 variabili calendario/Fourier)
COV_COLS = list(range(0, 18))          # 0..17  (18 covariate note sia nel passato che nel futuro)
HIST_TARGET_COLS = list(range(18, 21)) # 18..20 (3 indici satellitari passati)

# I 3 target da prevedere.
TARGET_NAMES = [
    "avg_NDVI_clear_sky",
    "avg_GNDVI_clear_sky",
    "avg_EVI_clear_sky",
]

# Classi climatiche di Köppen presenti nel dataset ("unknown" all'indice 0 per gestire ood-s).
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