# RenAge ensemble

RenAge predicts chronological age from DNA methylation beta values. This
repository contains only the software and frozen assets needed for inference.
It does not contain development scripts, methylation datasets, or sample-level
benchmark records.

## Requirements

- Linux or macOS
- Python 3.10 or newer
- CPU inference on both operating systems
- Optional NVIDIA CUDA on Linux or Apple Metal acceleration on macOS

## Installation

```bash
git clone https://github.com/GuoqinRen/RenAge.git
cd RenAge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
renage download
```

Install Parquet support when needed:

```bash
python -m pip install ".[parquet]"
```

## Input

Input files may be CSV, TSV, TXT, or Parquet. Beta values must be within
`0` and `1`. Both common orientations are accepted:

- one sample per row and one CpG per column;
- one CpG per row and one sample per column.

The first identifier column should contain sample identifiers or CpG
identifiers, as appropriate. Orientation is detected automatically, or it can
be specified explicitly. CpG identifiers are matched without regard to letter
case. Duplicate sample or CpG identifiers are rejected.

The predictor reports panel coverage and missing-value counts. By default, at
least 80% of required CpGs must be present. Use a different threshold only when
it is scientifically justified.

## Predict age

```bash
renage predict methylation.csv --output age_predictions.csv
```

Explicit orientation and device selection are also available:

```bash
renage predict methylation.tsv \
  --orientation cpg-by-sample \
  --device cpu \
  --output age_predictions.csv \
  --qc-output age_predictions.qc.json
```

The result contains two columns:

| Column | Meaning |
|---|---|
| `sample_id` | Input sample identifier |
| `predicted_age_years` | RenAge chronological-age prediction in years |

Run `renage predict --help` for all options. The default `auto` device uses
CUDA when available, then Apple Metal when available, and otherwise CPU.

## Performance summary

Mean absolute error (MAE) is reported in years. Comparisons use complete clock
pipelines on the same specimens within each benchmark.

### Established clocks

Equal-cohort results across three blood cohorts comprising 1,444 specimens:

| Clock | MAE (years) |
|---|---:|
| **RenAge ensemble** | **1.521** |
| Zhang BLUP | 2.686 |
| Skin & Blood | 2.856 |
| GP-age-30 | 3.952 |
| AltumAge | 4.469 |
| Horvath multi-tissue | 5.697 |

### Contemporary clocks

Identical-specimen, equal-cohort results across four blood cohorts comprising
556 specimens:

| Clock | MAE (years) |
|---|---:|
| **RenAge ensemble** | **2.175** |
| cAge | 2.696 |
| MAPLE | 4.105 |
| GT-Mamba | 5.648 |
| AltumAge | 5.772 |

DeepStrataAge could be evaluated in one exact-age cohort. In that cohort
(`n=388`), MAE was 1.287 years for the RenAge ensemble and 1.959 years for
DeepStrataAge.

### GSE111223

The cohort-specific RenAge ensemble result for GSE111223 is:

| Cohort | Samples | MAE (years) | RMSE (years) | R² |
|---|---:|---:|---:|---:|
| GSE111223 | 131 | 0.984 | 1.345 | 0.980 |

Results from different tables should not be pooled because their cohort sets
and sample counts differ.

## Reproducibility and asset integrity

`renage download` retrieves the versioned inference bundle from the GitHub
release and verifies its SHA-256 checksum. The bundle contains a single frozen
ensemble artifact, required CpG identifiers, frozen reference values, and a
manifest with per-file checksums. By default it is stored under the operating
system's user cache directory. Set `RENAGE_ASSET_DIR` or pass `--assets` to use
another location.

## Scope and limitations

RenAge is intended for research use. It is not a medical device and must not be
used by itself for diagnosis, treatment decisions, or individual clinical
interpretation. Performance can vary with tissue, assay platform, preprocessing,
cohort composition, and CpG coverage. Predictions outside the represented age
and tissue ranges require particular caution.

## Citation

Citation metadata is provided in `CITATION.cff`. Publication citation details
can be added when available.

## License

Copyright 2025, Guoqin Ren. See `LICENSE` for the repository's existing use and
distribution terms.
