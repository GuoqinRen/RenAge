"""Public inference interface for the RenAge ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .assets import resolve_assets, validate_asset_dir
from .input import InputQC, load_feature_ids, prepare_input


@dataclass(frozen=True)
class PredictionResult:
    predictions: pd.DataFrame
    qc: InputQC
    device: str
    asset_dir: Path


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("Apple Metal was requested but is not available")
    return device


def predict_matrix(
    matrix: np.ndarray,
    model_path: Path,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, str]:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("The aligned methylation matrix must be two-dimensional")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    runtime_device = select_device(device)
    module = torch.jit.load(str(model_path), map_location=runtime_device)
    module.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            tensor = torch.as_tensor(
                matrix[start : start + batch_size], dtype=torch.float32, device=runtime_device
            )
            predicted = module(tensor).detach().float().cpu().numpy().reshape(-1)
            chunks.append(predicted)
    values = np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0, dtype=np.float64)
    return values, runtime_device.type


def predict_file(
    input_path: str | Path,
    assets: str | Path | None = None,
    orientation: str = "auto",
    sample_id_column: str | None = None,
    min_coverage: float = 0.80,
    batch_size: int = 128,
    device: str = "auto",
    allow_download: bool = True,
) -> PredictionResult:
    asset_dir = resolve_assets(assets, allow_download=allow_download)
    validate_asset_dir(asset_dir)
    feature_ids = load_feature_ids(asset_dir / "feature_ids.txt")
    reference_values = np.load(asset_dir / "reference_values.npy", allow_pickle=False).astype(np.float32)
    sample_ids, matrix, qc = prepare_input(
        input_path,
        feature_ids,
        reference_values,
        orientation=orientation,
        sample_id_column=sample_id_column,
        min_coverage=min_coverage,
    )
    ages, device_name = predict_matrix(
        matrix,
        asset_dir / "renage_ensemble.pt",
        batch_size=batch_size,
        device=device,
    )
    predictions = pd.DataFrame({"sample_id": sample_ids, "predicted_age_years": ages})
    return PredictionResult(predictions=predictions, qc=qc, device=device_name, asset_dir=asset_dir)
