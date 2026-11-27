from pathlib import Path
from typing import Literal, Union

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

SplitName = Literal["train", "train_val", "val_id", "ood_s", "ood_t", "custom"]


class GreenEarthNetDataset(Dataset):
    """
    Dataset PyTorch per GreenEarthNet (previsione multi-output di NDVI, GNDVI, EVI).
    Carica i file .npz, applica lo split train/train_val senza data leakage
    e preprocessa ogni finestra su richiesta.
    """

    def __init__(
        self,
        split: SplitName = "train",
        data_dir: Union[str, Path, None] = None,
        scaler: ArcsinhScaler | None = None,
        val_ratio: float = 0.15,
        overlap_buffer: int = 6,
        climate_dropout_prob: float = 0.25,
    ):
        super().__init__()
        self.split = split
        self.scaler = scaler if scaler is not None else ArcsinhScaler(TRAIN_SCALER_PATH)
        self.is_training = split == "train"
        self.climate_dropout_prob = climate_dropout_prob if self.is_training else 0.0

        # Selezione della cartella dati corretta
        if data_dir is not None:
            folder = Path(data_dir)
        elif split in ("train", "train_val"):
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

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as z:
                n_total = len(z["target"])

                # Divisione temporale per area tra train (85%) e train_val (15%)
                # con buffer di sicurezza per evitare finestre sovrapposte al confine
                if split == "train":
                    cut = int(n_total * (1.0 - val_ratio))
                    indices = range(0, max(1, cut))
                elif split == "train_val":
                    cut = int(n_total * (1.0 - val_ratio)) + overlap_buffer
                    if cut >= n_total:
                        continue
                    indices = range(cut, n_total)
                else:
                    indices = range(0, n_total)

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

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = preprocess_window(
            history=self.histories[idx],
            future=self.futures[idx],
            target=self.targets[idx],
            future_target_positions=self.ftps[idx],
            climate=self.climates[idx],
            scaler=self.scaler,
        )

        # Climate dropout in training: abitua la rete a gestire climate="unknown" (indice 0) come in ood-s
        if self.climate_dropout_prob > 0.0 and np.random.rand() < self.climate_dropout_prob:
            item["climate_idx"] = np.int64(0)

        return item


def collate_greenearthnet(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
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

    # Maschere di padding: True dove il passo temporale è valido, False dove è padding
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
        "target_norm": torch.from_numpy(np.stack([s["target_norm"] for s in batch])),              # (B, 3, 3)
        "target_raw": torch.from_numpy(np.stack([s["target_raw"] for s in batch])),                # (B, 3, 3)
    }


def create_dataloader(
    split: SplitName,
    batch_size: int = 128,
    num_workers: int = 0,
    data_dir: Union[str, Path, None] = None,
    scaler: ArcsinhScaler | None = None,
) -> DataLoader:
    """Crea il DataLoader pronto per il training o la valutazione."""
    dataset = GreenEarthNetDataset(split=split, data_dir=data_dir, scaler=scaler)
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