import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from linear_regression.data import RegressionDataset, make_validation_split
from linear_regression.experiments.robustness.run import (
    noisy_inputs,
    run_study,
    sample_conditions,
)


class MeanRegressor:
    def __init__(self, fits):
        self.fits = fits

    def fit(self, X, y):
        self.fits.append(X[:, 0].copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class RobustnessTests(unittest.TestCase):
    def test_nested_subsets_keep_full_training_order(self):
        conditions = sample_conditions(20, fractions=(.1, .5, 1.), seeds=(1, 2))
        self.assertEqual(len(conditions), 5)
        self.assertTrue(set(conditions[0][2]) <= set(conditions[1][2]))
        np.testing.assert_array_equal(conditions[2][2], np.arange(20))
        self.assertEqual(len({tuple(indices) for _, _, indices in conditions}), 5)

    def test_noise_does_not_modify_validation_inputs(self):
        X = np.ones((2, 39))
        perturbed = noisy_inputs(X, np.full(39, 2.), .1, np.ones_like(X))
        np.testing.assert_allclose(perturbed, 1.2)
        np.testing.assert_array_equal(X, np.ones_like(X))

    def test_study_shares_conditions_and_writes_aligned_outputs(self):
        X = np.repeat(np.arange(30.)[:, None], 39, axis=1)
        data = RegressionDataset(X, np.arange(30.) / 100 + 1.2, "raw.csv")
        split = make_validation_split(data)
        fits = {"linear": [], "tree": []}
        builders = {name: (lambda variant, name=name: MeanRegressor(fits[name]))
                    for name in fits}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = run_study(split, output, builders=builders,
                                fractions=(.1, .5, 1.), subset_seeds=(1, 2),
                                noise_levels=(0., .05), noise_seeds=(10, 11))
            self.assertEqual(len(records), 16)
            self.assertEqual(len(fits["linear"]), 5)
            for left, right in zip(fits["linear"], fits["tree"]):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(fits["linear"][2], split.X_train[:, 0])
            self.assertTrue(all(row["evaluation_ids"] == list(split.evaluation_ids)
                                for row in records))
            self.assertTrue(all(set(row["train_ids"]).isdisjoint(row["evaluation_ids"])
                                for row in records if row["experiment"] == "sample_size"))
            with (output / "results.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(records))
            for filename in ("sample_size.png", "input_noise.png"):
                self.assertEqual((output / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((output / "protocol.json").exists())
            protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["train_rows"], len(split.train_ids))
            self.assertEqual(len(protocol["train_ids_sha256"]), 64)
            self.assertNotIn("train_ids", protocol)
            np.testing.assert_array_equal(split.X_eval, data.X[
                [int(row.rsplit(":", 1)[1]) - 1 for row in split.evaluation_ids]])


if __name__ == "__main__":
    unittest.main()
