"""Command-line interface for RenAge epigenetic age inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .assets import download_assets
from .predictor import predict_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="renage", description="RenAge epigenetic age inference")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download and verify the inference bundle")
    download.add_argument("--assets", type=Path, help="destination directory")
    download.add_argument("--force", action="store_true", help="replace an existing asset directory")

    predict = commands.add_parser("predict", help="predict age from a methylation matrix")
    predict.add_argument("input", type=Path, help="CSV, TSV, TXT, or Parquet methylation matrix")
    predict.add_argument("--output", type=Path, required=True, help="output CSV path")
    predict.add_argument("--qc-output", type=Path, help="optional QC JSON path")
    predict.add_argument("--assets", type=Path, help="inference asset directory")
    predict.add_argument(
        "--orientation",
        choices=("auto", "sample-by-cpg", "cpg-by-sample"),
        default="auto",
    )
    predict.add_argument("--sample-id-column", help="sample identifier column for sample-by-CpG input")
    predict.add_argument("--min-coverage", type=float, default=0.80)
    predict.add_argument("--batch-size", type=int, default=128)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict.add_argument("--no-download", action="store_true", help="fail instead of downloading missing assets")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "download":
        location = download_assets(args.assets, force=args.force)
        print(f"Verified RenAge assets: {location}")
        return

    result = predict_file(
        args.input,
        assets=args.assets,
        orientation=args.orientation,
        sample_id_column=args.sample_id_column,
        min_coverage=args.min_coverage,
        batch_size=args.batch_size,
        device=args.device,
        allow_download=not args.no_download,
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output, index=False)
    qc_payload = {**result.qc.to_dict(), "device": result.device, "asset_dir": str(result.asset_dir)}
    if args.qc_output:
        args.qc_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.qc_output.write_text(json.dumps(qc_payload, indent=2) + "\n")
    print(
        f"Predicted {len(result.predictions)} samples on {result.device}; "
        f"CpG coverage {result.qc.coverage_fraction:.1%}; output {args.output}"
    )


if __name__ == "__main__":
    main()
