from pathlib import Path
import unittest

from renage.cli import _build_parser


class CliTests(unittest.TestCase):
    def test_actual_age_field_option(self) -> None:
        args = _build_parser().parse_args(
            [
                "predict",
                "methylation.csv",
                "--output",
                "predictions.csv",
                "--actual-age-field",
                "Age",
            ]
        )
        self.assertEqual(args.actual_age_field, "Age")
        self.assertEqual(args.input, Path("methylation.csv"))

    def test_age_column_alias(self) -> None:
        args = _build_parser().parse_args(
            [
                "predict",
                "methylation.csv",
                "--output",
                "predictions.csv",
                "--age-column",
                "Age",
            ]
        )
        self.assertEqual(args.actual_age_field, "Age")


if __name__ == "__main__":
    unittest.main()
