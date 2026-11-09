"""Check the numerical evidence without treating statistics as proof of roles."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from linear_regression.experiments.data.column_roles import (
    read_raw_matrix,
    summarize_columns,
)


class ColumnRoleTests(unittest.TestCase):
    def test_read_preserves_all_columns_and_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            values = np.arange(82).reshape(2, 41)
            np.savetxt(path, values)
            np.testing.assert_array_equal(read_raw_matrix(path), values)
            for content in ["", "1 2\n", "bad " + "0 " * 40,
                            "nan " + "0 " * 40, "inf " + "0 " * 40]:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content[:12]):
                    with self.assertRaises(ValueError):
                        read_raw_matrix(path)

    def test_evidence_has_known_cdf_distance_and_excludes_self_correlation(self):
        # Uniform quantile midpoints have exactly 1/(2*n) KS distance.
        n = 1000
        midpoints = 0.7 + 4.3 * (np.arange(n) + 0.5) / n
        matrix = np.tile(midpoints[:, None], (1, 41))
        self.assertAlmostEqual(summarize_columns(matrix)[2]["uniform_ks_distance"], 1 / (2 * n))

        # Orthogonal input signals: each has zero correlation with the OTHER
        # inputs, but a nonzero correlation if accidentally included in the mean.
        phase = 2 * np.pi * np.arange(n) / n
        features = 2.85 + np.sin(phase[:, None] * np.arange(1, 40))
        mean = features.mean(axis=1)
        matrix = np.column_stack((1.3 + mean / 100, 2 - mean / 100, features))
        records = summarize_columns(matrix)
        self.assertAlmostEqual(records[0]["corr_with_other_feature_mean"], 1)
        self.assertAlmostEqual(records[1]["corr_with_other_feature_mean"], -1)
        for record in records[2:]:
            self.assertAlmostEqual(record["corr_with_other_feature_mean"], 0)


if __name__ == "__main__":
    unittest.main()
