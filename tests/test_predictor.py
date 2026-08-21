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


if __name__ == "__main__":
    unittest.main()
