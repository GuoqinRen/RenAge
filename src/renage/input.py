"""Read, validate, and align DNA methylation beta-value matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ORIENTATIONS = ("auto", "sample-by-cpg", "cpg-by-sample")


@dataclass(frozen=True)
class InputQC:
    source_path: str
    orientation: str
    samples: int
    required_cpgs: int
    present_cpgs: int
    coverage_fraction: float
    absent_cpgs: int
    missing_input_values: int
    cohort_reference_imputations: int
    frozen_reference_imputations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_feature_ids(path: Path) -> list[str]:
    values = [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("The required CpG identifier list is empty or contains duplicates")
    return values


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Input must be CSV, TSV, TXT, or Parquet")


def _infer_orientation(frame: pd.DataFrame) -> str:
    header_cpgs = sum(str(column).strip().lower().startswith("cg") for column in frame.columns)
    first_values = frame.iloc[:, 0].astype(str).str.strip().str.lower() if len(frame.columns) else pd.Series(dtype=str)
    row_cpgs = int(first_values.str.startswith("cg").sum())
    if header_cpgs >= max(3, len(frame.columns) // 2):
        return "sample-by-cpg"
    if row_cpgs >= max(3, len(frame) // 2):
        return "cpg-by-sample"
    raise ValueError("Could not infer matrix orientation; pass --orientation explicitly")


def _normalize_sample_by_cpg(
    frame: pd.DataFrame,
    sample_id_column: str | None,
) -> tuple[list[str], pd.DataFrame]:
    if sample_id_column:
        if sample_id_column not in frame.columns:
            raise ValueError(f"Sample identifier column {sample_id_column!r} was not found")
        identifier_column = sample_id_column
    else:
        non_cpg = [column for column in frame.columns if not str(column).strip().lower().startswith("cg")]
        if not non_cpg:
            raise ValueError("A sample identifier column is required")
        identifier_column = non_cpg[0]
    sample_ids = frame[identifier_column].astype(str).str.strip().tolist()
    values = frame.drop(columns=[identifier_column])
    values.columns = [str(column).strip().lower() for column in values.columns]
    values = values.loc[:, [column.startswith("cg") for column in values.columns]]
    return sample_ids, values


def _normalize_cpg_by_sample(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("A CpG-by-sample matrix requires an identifier column and at least one sample")
    identifier_column = frame.columns[0]
    cpg_ids = frame[identifier_column].astype(str).str.strip().str.lower()
    if cpg_ids.duplicated().any():
        raise ValueError("Duplicate CpG identifiers are not allowed")
    values = frame.drop(columns=[identifier_column]).copy()
    values.index = cpg_ids
    values = values.transpose()
    sample_ids = [str(value).strip() for value in values.index]
    values.columns = [str(value).strip().lower() for value in values.columns]
    values.reset_index(drop=True, inplace=True)
    return sample_ids, values


def _extract_actual_ages(
    frame: pd.DataFrame,
    orientation: str,
    actual_age_field: str | None,
    sample_ids: list[str],
) -> np.ndarray | None:
    if actual_age_field is None:
        return None
    requested = str(actual_age_field).strip().casefold()
    if not requested:
        raise ValueError("Actual age field must be non-empty")

    if orientation == "sample-by-cpg":
        matches = [
            column
            for column in frame.columns
            if str(column).strip().casefold() == requested
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Actual age column {actual_age_field!r} was not found uniquely"
            )
        raw = frame[matches[0]]
    else:
        identifier_column = frame.columns[0]
        labels = frame[identifier_column].astype(str).str.strip().str.casefold()
        positions = np.flatnonzero(labels.to_numpy() == requested)
        if len(positions) != 1:
            raise ValueError(
                f"Actual age row {actual_age_field!r} was not found uniquely"
            )
        matrix_sample_ids = [str(value).strip() for value in frame.columns[1:]]
        if matrix_sample_ids != sample_ids:
            raise RuntimeError("Actual ages could not be aligned to matrix sample identifiers")
        raw = frame.iloc[int(positions[0]), 1:]

    numeric = pd.to_numeric(raw, errors="coerce")
    invalid_text = raw.notna() & numeric.isna()
    if bool(invalid_text.to_numpy().any()):
        raise ValueError("Actual ages must be numeric or missing")
    ages = numeric.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(ages).any():
        raise ValueError("Actual ages must be finite or missing")
    finite = np.isfinite(ages)
    if finite.any() and (ages[finite] < 0).any():
        raise ValueError("Actual ages must be non-negative")
    return ages


def _prepare_input(
    path: str | Path,
    feature_ids: list[str],
    reference_values: np.ndarray,
    orientation: str = "auto",
    sample_id_column: str | None = None,
    min_coverage: float = 0.80,
    actual_age_field: str | None = None,
) -> tuple[list[str], np.ndarray, InputQC, np.ndarray | None, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation {orientation!r}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("Minimum coverage must be between 0 and 1")
    if reference_values.shape != (len(feature_ids),) or not np.isfinite(reference_values).all():
        raise ValueError("Frozen reference values do not match the required CpG identifiers")

    frame = _read_frame(source)
    resolved_orientation = _infer_orientation(frame) if orientation == "auto" else orientation
    if resolved_orientation == "sample-by-cpg":
        sample_ids, values = _normalize_sample_by_cpg(frame, sample_id_column)
    else:
        sample_ids, values = _normalize_cpg_by_sample(frame)

    actual_ages = _extract_actual_ages(
        frame,
        resolved_orientation,
        actual_age_field,
        sample_ids,
    )

    if not sample_ids or any(not value for value in sample_ids):
        raise ValueError("Sample identifiers must be non-empty")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample identifiers are not allowed")
    if values.columns.duplicated().any():
        raise ValueError("Duplicate CpG identifiers are not allowed")

    feature_positions = {feature: index for index, feature in enumerate(feature_ids)}
    present_ids = [feature for feature in feature_ids if feature in values.columns]
    coverage = len(present_ids) / len(feature_ids)
    if coverage < min_coverage:
        raise ValueError(
            f"Required CpG coverage is {coverage:.1%}, below the minimum of {min_coverage:.1%}"
        )

    selected = values.loc[:, present_ids]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    invalid_text = selected.notna() & numeric.isna()
    if bool(invalid_text.to_numpy().any()):
        raise ValueError("The methylation matrix contains non-numeric beta values")
    observed = numeric.to_numpy(dtype=np.float32, copy=True)
    if np.isinf(observed).any():
        raise ValueError("The methylation matrix contains infinite beta values")
    finite = np.isfinite(observed)
    if finite.any() and ((observed[finite] < 0).any() or (observed[finite] > 1).any()):
        raise ValueError("Beta values must be within 0 and 1")

    output = np.broadcast_to(reference_values.astype(np.float32), (len(sample_ids), len(feature_ids))).copy()
    positions = np.asarray([feature_positions[feature] for feature in present_ids], dtype=int)
    missing = ~finite
    missing_count = int(missing.sum())
    absent_count = len(feature_ids) - len(present_ids)
    missing_cpg_percentages = (
        (absent_count + missing.sum(axis=1)) / len(feature_ids) * 100.0
    )
    counts = finite.sum(axis=0)
    sums = np.where(finite, observed, 0.0).sum(axis=0, dtype=np.float64)
    cohort_values = np.full(len(present_ids), np.nan, dtype=np.float32)
    valid_columns = counts > 0
    cohort_values[valid_columns] = (sums[valid_columns] / counts[valid_columns]).astype(np.float32)
    rows, columns = np.where(missing)
    cohort_imputations = 0
    frozen_imputations = len(sample_ids) * (len(feature_ids) - len(present_ids))
    if len(rows):
        replacements = cohort_values[columns]
        use_frozen = ~np.isfinite(replacements)
        observed[rows[~use_frozen], columns[~use_frozen]] = replacements[~use_frozen]
        cohort_imputations = int((~use_frozen).sum())
        observed[rows[use_frozen], columns[use_frozen]] = reference_values[positions[columns[use_frozen]]]
        frozen_imputations += int(use_frozen.sum())
    output[:, positions] = observed

    qc = InputQC(
        source_path=str(source),
        orientation=resolved_orientation,
        samples=len(sample_ids),
        required_cpgs=len(feature_ids),
        present_cpgs=len(present_ids),
        coverage_fraction=coverage,
        absent_cpgs=absent_count,
        missing_input_values=missing_count,
        cohort_reference_imputations=cohort_imputations,
        frozen_reference_imputations=frozen_imputations,
    )
    return sample_ids, output, qc, actual_ages, missing_cpg_percentages


def prepare_input(
    path: str | Path,
    feature_ids: list[str],
    reference_values: np.ndarray,
    orientation: str = "auto",
    sample_id_column: str | None = None,
    min_coverage: float = 0.80,
) -> tuple[list[str], np.ndarray, InputQC]:
    sample_ids, matrix, qc, _, _ = _prepare_input(
        path,
        feature_ids,
        reference_values,
        orientation=orientation,
        sample_id_column=sample_id_column,
        min_coverage=min_coverage,
    )
    return sample_ids, matrix, qc


def prepare_input_with_actual_ages(
    path: str | Path,
    feature_ids: list[str],
    reference_values: np.ndarray,
    orientation: str = "auto",
    sample_id_column: str | None = None,
    min_coverage: float = 0.80,
    actual_age_field: str | None = None,
) -> tuple[list[str], np.ndarray, InputQC, np.ndarray | None]:
    sample_ids, matrix, qc, actual_ages, _ = _prepare_input(
        path,
        feature_ids,
        reference_values,
        orientation=orientation,
        sample_id_column=sample_id_column,
        min_coverage=min_coverage,
        actual_age_field=actual_age_field,
    )
    return sample_ids, matrix, qc, actual_ages


def prepare_prediction_input(
    path: str | Path,
    feature_ids: list[str],
    reference_values: np.ndarray,
    orientation: str = "auto",
    sample_id_column: str | None = None,
    min_coverage: float = 0.80,
    actual_age_field: str | None = None,
) -> tuple[list[str], np.ndarray, InputQC, np.ndarray | None, np.ndarray]:
    """Prepare inference input and return per-sample missing-CpG percentages."""
    return _prepare_input(
        path,
        feature_ids,
        reference_values,
        orientation=orientation,
        sample_id_column=sample_id_column,
        min_coverage=min_coverage,
        actual_age_field=actual_age_field,
    )
