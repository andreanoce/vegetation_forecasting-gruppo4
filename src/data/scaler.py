import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.config import TRAIN_SCALER_PATH

ArrayOrTensor = Union[np.ndarray, torch.Tensor]


class ArcsinhScaler:
    """
    Implementa la normalizzazione ufficiale 'standardization_arcsinh' usando
    esclusivamente le statistiche calcolate sullo split di train (scaler.json):
        1. z = (x - mean) / sqrt(var + eps)
        2. y = arcsinh(z)

    Inversa (per riportare target e predizioni dei quantili alla scala originale):
        x = sinh(y) * sqrt(var + eps) + mean
    """

    def __init__(self, scaler_path: Union[str, Path] = TRAIN_SCALER_PATH):
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"File scaler.json non trovato in: {scaler_path}")

        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.mode = data.get("mode", "standardization_arcsinh")
        self.eps = float(data.get("eps", 1e-5))

        # Statistiche delle 21 feature: shape (21,)
        self.feature_mean = np.array(data["mean"], dtype=np.float32)
        self.feature_var = np.array(data["var"], dtype=np.float32)
        self.feature_std = np.sqrt(self.feature_var + self.eps).astype(np.float32)

        # Statistiche dei 3 target (NDVI, GNDVI, EVI): shape (3,)
        self.target_mean = np.array(data["target_mean"], dtype=np.float32)
        self.target_var = np.array(data["target_var"], dtype=np.float32)
        self.target_std = np.sqrt(self.target_var + self.eps).astype(np.float32)

    def transform_features(self, x: np.ndarray, cols: list[int] | None = None) -> np.ndarray:
        """
        Normalizza le feature (es. shape (T, 21) o (T, 18) se cols=range(18)).
        I valori NaN rimangono NaN senza generare warning.
        """
        x = np.asarray(x, dtype=np.float32)
        mean = self.feature_mean if cols is None else self.feature_mean[cols]
        std = self.feature_std if cols is None else self.feature_std[cols]

        z = (x - mean) / std
        return np.arcsinh(z)

    def inverse_transform_features(self, y: np.ndarray, cols: list[int] | None = None) -> np.ndarray:
        """Inverte la normalizzazione sulle feature."""
        y = np.asarray(y, dtype=np.float32)
        mean = self.feature_mean if cols is None else self.feature_mean[cols]
        std = self.feature_std if cols is None else self.feature_std[cols]

        return np.sinh(y) * std + mean

    def transform_target(self, target: ArrayOrTensor) -> ArrayOrTensor:
        """
        Normalizza i target (NDVI, GNDVI, EVI) sull'ultima dimensione (..., 3).
        Accetta NumPy array o PyTorch Tensor.
        """
        if isinstance(target, torch.Tensor):
            mean = torch.as_tensor(self.target_mean, dtype=target.dtype, device=target.device)
            std = torch.as_tensor(self.target_std, dtype=target.dtype, device=target.device)
            return torch.arcsinh((target - mean) / std)

        target = np.asarray(target, dtype=np.float32)
        z = (target - self.target_mean) / self.target_std
        return np.arcsinh(z)

    def inverse_transform_target(
        self,
        y: ArrayOrTensor,
        target_axis: int | None = None,
    ) -> ArrayOrTensor:
        """
        Inverte la normalizzazione sui target o sulle predizioni dei quantili.
        Convenzione shape del progetto:
          - Target senza quantili: (3_step, 3_indici) o (Batch, 3_step, 3_indici) -> target_axis = -1
          - Predizioni coi 3 quantili: (Batch, 3_step, 3_indici, 3_quantili)      -> target_axis = -2
        Per un singolo campione 3D coi quantili (3_step, 3_indici, 3_quantili), passare target_axis=-2.
        """
        if not isinstance(y, torch.Tensor):
            y = np.asarray(y, dtype=np.float32)

        if target_axis is None:
            target_axis = -2 if y.ndim == 4 else -1

        if y.shape[target_axis] != 3:
            raise ValueError(
                f"Attesa dimensione 3 su target_axis={target_axis}, trovata shape {tuple(y.shape)}"
            )

        shape = [1] * y.ndim
        shape[target_axis] = 3

        if isinstance(y, torch.Tensor):
            mean = torch.as_tensor(self.target_mean, dtype=y.dtype, device=y.device).view(*shape)
            std = torch.as_tensor(self.target_std, dtype=y.dtype, device=y.device).view(*shape)
            return torch.sinh(y) * std + mean

        mean = self.target_mean.reshape(*shape)
        std = self.target_std.reshape(*shape)
        return np.sinh(y) * std + mean