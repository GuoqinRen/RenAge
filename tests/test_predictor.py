import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from renage.predictor import predict_file


class MeanModule(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PredictorTests(unittest.TestCase):
    def test_predict_file_with_local_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            (assets / "feature_ids.txt").write_text("cg0001\ncg0002\ncg0003\n")
            np.save(assets / "reference_values.npy", np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
            traced = torch.jit.trace(MeanModule().eval(), torch.zeros((1, 3), dtype=torch.float32))
            traced.save(str(assets / "renage_ensemble.pt"))
            files = {}
            for name in ("renage_ensemble.pt", "feature_ids.txt", "reference_values.npy"):
                path = assets / name
                files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
            (assets / "manifest.json").write_text(
                json.dumps({"asset_version": "1.0.0", "files": files}, indent=2) + "\n"
            )

            source = root / "input.csv"
            pd.DataFrame(
                {
                    "sample_id": ["one", "two"],
                    "Age": [42.5, 67.0],
                    "cg0001": [0.1, 0.4],
                    "cg0002": [0.2, 0.5],
                    "cg0003": [0.3, 0.6],
                }
            ).to_csv(source, index=False)
            result = predict_file(source, assets=assets, allow_download=False, device="cpu")
            self.assertEqual(result.predictions["sample_id"].tolist(), ["one", "two"])
            np.testing.assert_allclose(
                result.predictions["predicted_age_years"], [0.2, 0.5], rtol=1e-6
            )
            self.assertEqual(
                result.predictions.columns.tolist(),
                ["sample_id", "predicted_age_years", "missing_cpg_percentage"],
            )
            np.testing.assert_allclose(result.predictions["missing_cpg_percentage"], [0.0, 0.0])

            comparison = predict_file(
                source,
                assets=assets,
                actual_age_field="Age",
                allow_download=False,
                device="cpu",
            )
            self.assertEqual(
                comparison.predictions.columns.tolist(),
                [
                    "sample_id",
                    "predicted_age_years",
                    "actual_age_years",
                    "missing_cpg_percentage",
                ],
            )
            np.testing.assert_allclose(comparison.predictions["actual_age_years"], [42.5, 67.0])
            np.testing.assert_allclose(
                comparison.predictions["predicted_age_years"], [0.2, 0.5], rtol=1e-6
            )

    def test_missing_cpg_percentage_includes_absent_and_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            (assets / "feature_ids.txt").write_text("cg0001\ncg0002\ncg0003\ncg0004\n")
            np.save(
                assets / "reference_values.npy",
                np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            )
            traced = torch.jit.trace(MeanModule().eval(), torch.zeros((1, 4), dtype=torch.float32))
            traced.save(str(assets / "renage_ensemble.pt"))
            files = {}
            for name in ("renage_ensemble.pt", "feature_ids.txt", "reference_values.npy"):
                path = assets / name
                files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
            (assets / "manifest.json").write_text(
                json.dumps({"asset_version": "1.0.0", "files": files}, indent=2) + "\n"
            )

            source = root / "input.csv"
            pd.DataFrame(
                {
                    "sample_id": ["complete", "one-empty"],
                    "cg0001": [0.1, np.nan],
                    "cg0002": [0.2, 0.2],
                    "cg0003": [0.3, 0.3],
                }
            ).to_csv(source, index=False)
            result = predict_file(
                source,
                assets=assets,
                min_coverage=0.75,
                allow_download=False,
                device="cpu",
            )

            np.testing.assert_allclose(
                result.predictions["missing_cpg_percentage"],
                [25.0, 50.0],
            )


if __name__ == "__main__":
    unittest.main()
