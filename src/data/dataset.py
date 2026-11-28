from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    OOD_S_DIR,
    OOD_T_DIR,
    SEED,
    TRAIN_DIR,
    TRAIN_SCALER_PATH,
    VAL_ID_DIR,
)
from src.data.preprocessing import preprocess_window
from src.data.scaler import ArcsinhScaler

SplitName = Literal[
    "train",
    "train_val",
    "train_val_s",
    "train_val_t",
    "val_id",
    "ood_s",
    "ood_t",
    "custom",
]

# 8 aree stratificate per clima riservate alla validazione spaziale (coprono BSk, Csa, Csb, Cfa, Cfb, Dfb, Dfc, unknown
# senza svuotare nessuna classe rara come Dfa dal training set)
SPATIAL_VAL_AREAS = {
    "29TPH.npz",  # Csb
    "30TYR.npz",  # Cfb
    "31TDH.npz",  # Csa
    "32TNR.npz",  # Cfa
    "32VMM.npz",  # Dfc
    "32VPP.npz",  # unknown
    "34TDQ.npz",  # Dfb
    "34TEL.npz",  # BSk
}


class GreenEarthNetDataset(Dataset):
    """
    Dataset PyTorch per GreenEarthNet (previsione multi-output di NDVI, GNDVI, EVI).
    Sullo split di train applica una divisione Spazio-Temporale basata sulle date reali:
      - Validazione Spaziale ('train_val_s'): 8 aree stratificate per clima mai viste in 'train'.
      - Validazione Temporale ('train_val_t'): sulle restanti 76 aree, tutte le finestre con
        primo target futuro >= '2019-08-15', mentre 'train' usa solo finestre con ultimo target
        futuro < '2019-07-15' (buffer di 31 giorni di calendario -> zero sovrapposizioni).
      - 'train_val': unione di 'train_val_s' e 'train_val_t' (con flag booleano 'is_spatial_val').
    """

    def __init__(
        self,
        split: SplitName = "train",
        data_dir: Union[str, Path, None] = None,
        scaler: ArcsinhScaler | None = None,
        train_cutoff_date: str = "2019-07-15",
        val_start_date: str = "2019-08-15",
        spatial_val_areas: set[str] | None = None,
        climate_dropout_prob: float = 0.25,
    ):
        super().__init__()
        self.split = split
        self.scaler = scaler if scaler is not None else ArcsinhScaler(TRAIN_SCALER_PATH)
        self.is_training = split == "train"
        self.climate_dropout_prob = climate_dropout_prob if self.is_training else 0.0
        held_out_areas = SPATIAL_VAL_AREAS if spatial_val_areas is None else spatial_val_areas

        if data_dir is not None:
            folder = Path(data_dir)
        elif split in ("train", "train_val", "train_val_s", "train_val_t"):
            folder = TRAIN_DIR
        elif split == "val_id":
            folder = VAL_ID_DIR
        elif split == "ood_s":
            folder = OOD_S_DIR
        elif split == "ood_t":
            folder = OOD_T_DIR
        else:
            raise ValueError(f"Split non riconosciuto: {split}")

        npz_files = sorted(folder.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"Nessun file .npz trovato in: {folder}")

        self.histories: list[np.ndarray] = []
        self.futures: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.ftps: list[np.ndarray] = []
        self.climates: list[str] = []
        self.is_spatial_val: list[bool] = []

        for npz_path in npz_files:
            is_spatial_area = npz_path.name in held_out_areas

            if split == "train" and is_spatial_area:
                continue
            if split == "train_val_s" and not is_spatial_area:
                continue
            if split == "train_val_t" and is_spatial_area:
                continue

            with np.load(npz_path, allow_pickle=True) as z:
                n_total = len(z["target"])

                if split == "train":
                    tts = z["target_timestamps"]
                    indices = [
                        i for i in range(n_total)
                        if str(tts[i][-1])[:10] < train_cutoff_date
                    ]
                elif split == "train_val_s" or (split == "train_val" and is_spatial_area):
                    indices = list(range(n_total))
                elif split in ("train_val", "train_val_t") and not is_spatial_area:
                    tts = z["target_timestamps"]
                    indices = [
                        i for i in range(n_total)
                        if str(tts[i][0])[:10] >= val_start_date
                    ]
                else:
                    indices = list(range(n_total))

                if not indices:
                    continue

                hist_arr = z["history"]
                fut_arr = z["future"]
                targ_arr = z["target"]
                ftp_arr = z["future_target_positions"]
                clim_arr = z["climate"]

                for i in indices:
                    self.histories.append(hist_arr[i])
                    self.futures.append(fut_arr[i])
                    self.targets.append(targ_arr[i])
                    self.ftps.append(ftp_arr[i])
                    self.climates.append(str(clim_arr[i]))
                    self.is_spatial_val.append(bool(is_spatial_area))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = preprocess_window(
            history=self.histories[idx],
            future=self.futures[idx],
            target=self.targets[idx],
            future_target_positions=self.ftps[idx],
            climate=self.climates[idx],
            scaler=self.scaler,
        )

        if self.climate_dropout_prob > 0.0 and torch.rand(1).item() < self.climate_dropout_prob:
            item["climate_idx"] = np.int64(0)

        item["is_spatial_val"] = self.is_spatial_val[idx]
        return item


def collate_greenearthnet(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Unisce una lista di campioni di lunghezza variabile in un batch PyTorch,
    applicando zero-padding a history_seq e future_seq e creando le maschere booleane.
    """
    batch_size = len(batch)
    h_lens = [sample["history_seq"].shape[0] for sample in batch]
    f_lens = [sample["future_seq"].shape[0] for sample in batch]

    max_h = max(h_lens)
    max_f = max(f_lens)
    h_dim = batch[0]["history_seq"].shape[1]
    f_dim = batch[0]["future_seq"].shape[1]

    history_padded = torch.zeros((batch_size, max_h, h_dim), dtype=torch.float32)
    future_padded = torch.zeros((batch_size, max_f, f_dim), dtype=torch.float32)

    history_valid_mask = torch.zeros((batch_size, max_h), dtype=torch.bool)
    future_valid_mask = torch.zeros((batch_size, max_f), dtype=torch.bool)

    for i, sample in enumerate(batch):
        hl = h_lens[i]
        fl = f_lens[i]
        history_padded[i, :hl] = torch.from_numpy(sample["history_seq"])
        future_padded[i, :fl] = torch.from_numpy(sample["future_seq"])
        history_valid_mask[i, :hl] = True
        future_valid_mask[i, :fl] = True

    return {
        "history_seq": history_padded,                                                             # (B, max_H, 24)
        "history_lengths": torch.tensor(h_lens, dtype=torch.long),                                 # (B,)
        "history_valid_mask": history_valid_mask,                                                  # (B, max_H)
        "future_seq": future_padded,                                                               # (B, max_F, 21)
        "future_lengths": torch.tensor(f_lens, dtype=torch.long),                                  # (B,)
        "future_valid_mask": future_valid_mask,                                                    # (B, max_F)
        "hist_clear_steps": torch.from_numpy(np.stack([s["hist_clear_steps"] for s in batch])),    # (B, 3, 22)
        "future_step_features": torch.from_numpy(np.stack([s["future_step_features"] for s in batch])),  # (B, 3, 27)
        "future_target_positions": torch.from_numpy(np.stack([s["future_target_positions"] for s in batch])),  # (B, 3)
        "last_clear_target_norm": torch.from_numpy(np.stack([s["last_clear_target_norm"] for s in batch])),    # (B, 3)
        "last_clear_target_raw": torch.from_numpy(np.stack([s["last_clear_target_raw"] for s in batch])),      # (B, 3)
        "climate_idx": torch.tensor([s["climate_idx"] for s in batch], dtype=torch.long),          # (B,)
        "is_spatial_val": torch.tensor([s["is_spatial_val"] for s in batch], dtype=torch.bool),    # (B,)
        "target_norm": torch.from_numpy(np.stack([s["target_norm"] for s in batch])),              # (B, 3, 3)
        "target_raw": torch.from_numpy(np.stack([s["target_raw"] for s in batch])),                # (B, 3, 3)
    }


def create_dataloader(
    split: SplitName,
    batch_size: int = 128,
    num_workers: int = 0,
    data_dir: Union[str, Path, None] = None,
    scaler: ArcsinhScaler | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Crea il DataLoader pronto per il training o la valutazione."""
    dataset = GreenEarthNetDataset(
        split=split,
        data_dir=data_dir,
        scaler=scaler,
        **dataset_kwargs,
    )
    shuffle = split == "train"
    generator = torch.Generator().manual_seed(SEED) if shuffle else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_greenearthnet,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )