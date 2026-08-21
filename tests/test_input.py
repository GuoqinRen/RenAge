from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from renage.input import prepare_input, prepare_input_with_actual_ages


FEATURES = ["cg0001", "cg0002", "cg0003"]
REFERENCE = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


class InputTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sample_by_cpg_alignment_and_imputation(self) -> None:
        source = self.root / "samples.csv"
        pd.DataFrame(
            {
                "sample_id": ["sample-a", "sample-b"],
                "CG0002": [np.nan, 0.4],
                "cg0001": [0.2, 0.6],
            }
        ).to_csv(source, index=False)
        sample_ids, matrix, qc = prepare_input(
            source,
            FEATURES,
            REFERENCE,
            orientation="sample-by-cpg",
            min_coverage=0.5,
        )
        self.assertEqual(sample_ids, ["sample-a", "sample-b"])
        np.testing.assert_allclose(matrix, [[0.2, 0.4, 0.3], [0.6, 0.4, 0.3]])
        self.assertEqual(qc.present_cpgs, 2)
        self.assertEqual(qc.cohort_reference_imputations, 1)
        self.assertEqual(qc.frozen_reference_imputations, 2)


    def test_cpg_by_sample_orientation(self) -> None:
        source = self.root / "cpgs.tsv"
        pd.DataFrame(
            {
                "cpg_id": ["cg0001", "cg0002", "cg0003"],
                "sample-a": [0.1, 0.2, 0.3],
                "sample-b": [0.4, 0.5, 0.6],
            }
        ).to_csv(source, sep="\t", index=False)
        sample_ids, matrix, qc = prepare_input(source, FEATURES, REFERENCE)
        self.assertEqual(sample_ids, ["sample-a", "sample-b"])
        np.testing.assert_allclose(matrix, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(qc.orientation, "cpg-by-sample")

    def test_actual_age_column_is_returned_in_sample_order(self) -> None:
        source = self.root / "samples_with_age.csv"
        pd.DataFrame(
            {
                "sample_id": ["sample-a", "sample-b"],
                "Age": [42.5, 67.0],
                "cg0001": [0.1, 0.4],
                "cg0002": [0.2, 0.5],
                "cg0003": [0.3, 0.6],
            }
        ).to_csv(source, index=False)
        sample_ids, matrix, _, actual_ages = prepare_input_with_actual_ages(
            source,
            FEATURES,
            REFERENCE,
            orientation="sample-by-cpg",
            actual_age_field="age",
        )
        self.assertEqual(sample_ids, ["sample-a", "sample-b"])
        np.testing.assert_allclose(matrix, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_allclose(actual_ages, [42.5, 67.0])

    def test_actual_age_row_is_returned_in_sample_order(self) -> None:
        source = self.root / "cpgs_with_age.tsv"
        pd.DataFrame(
            {
                "field": ["cg0001", "cg0002", "cg0003", "Age"],
                "sample-a": [0.1, 0.2, 0.3, 42.5],
                "sample-b": [0.4, 0.5, 0.6, 67.0],
            }
        ).to_csv(source, sep="\t", index=False)
        sample_ids, matrix, qc, actual_ages = prepare_input_with_actual_ages(
            source,
            FEATURES,
            REFERENCE,
            actual_age_field="Age",
        )
        self.assertEqual(sample_ids, ["sample-a", "sample-b"])
        np.testing.assert_allclose(matrix, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_allclose(actual_ages, [42.5, 67.0])
        self.assertEqual(qc.orientation, "cpg-by-sample")

    def test_invalid_actual_age_is_rejected(self) -> None:
        source = self.root / "invalid_age.csv"
        pd.DataFrame(
            {
                "sample_id": ["sample-a"],
                "Age": ["unknown"],
                "cg0001": [0.1],
                "cg0002": [0.2],
                "cg0003": [0.3],
            }
        ).to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "Actual ages must be numeric"):
            prepare_input_with_actual_ages(
                source,
                FEATURES,
                REFERENCE,
                orientation="sample-by-cpg",
                actual_age_field="Age",
            )

    def test_invalid_beta_value_is_rejected(self) -> None:
        source = self.root / "invalid.csv"
        pd.DataFrame(
            {"sample_id": ["sample-a"], "cg0001": [1.2], "cg0002": [0.2], "cg0003": [0.3]}
        ).to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "within 0 and 1"):
            prepare_input(source, FEATURES, REFERENCE, orientation="sample-by-cpg")


if __name__ == "__main__":
    unittest.main()
