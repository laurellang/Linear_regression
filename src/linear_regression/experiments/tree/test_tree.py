"""Small behavior checks for the final model; no full-data training."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from linear_regression.data import EvaluationSplit
from . import model, run


def settings():
    return {"model": "RandomForestRegressor", "parameters": {
        "n_estimators": 3, "random_state": 42, "n_jobs": 1}}


def example_split():
    return EvaluationSplit(
        np.repeat(np.arange(3.0)[:, None], 39, axis=1), np.array([1., 2., 3.]),
        np.repeat(np.array([5., 3., 4.])[:, None], 39, axis=1), np.array([4., 0., 2.]),
        "validation", ("row:5", "row:3", "row:4"), ("row:0", "row:1", "row:2"), 42)


class TreeTests(unittest.TestCase):
    def test_factory_is_unfitted_cloneable_and_independent_of_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            path.write_text(json.dumps(settings()), encoding="utf-8")
            with patch.object(model, "PARAMS_PATH", path):
                try:
                    os.chdir(directory)
                    first, second = model.build_model(), model.build_model("optimized")
                    self.assertIsInstance(first, Pipeline)
                    self.assertIsNot(first, second)
                    for estimator in (first, second, clone(first)):
                        with self.assertRaises(NotFittedError):
                            check_is_fitted(estimator)
                    for variant in ("baseline", "previous", "reference"):
                        with self.assertRaises(ValueError):
                            model.build_model(variant)
                finally:
                    os.chdir(previous)

    def test_mean_is_row_local_target_independent_and_preserves_input(self):
        X = np.arange(6 * 39, dtype=float).reshape(6, 39)
        original = X.copy()
        transform = model.AddMeanFeature().fit(X, np.zeros(6))
        actual = transform.transform(X)
        np.testing.assert_array_equal(actual[:, :39], original)
        np.testing.assert_array_equal(actual[:, 39], original.mean(axis=1))
        np.testing.assert_array_equal(transform.transform(X[:1]), actual[:1])
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(actual, model.AddMeanFeature().fit_transform(X, np.ones(6)))
        for invalid in (np.zeros((3, 40)), np.full((3, 39), np.nan)):
            with self.assertRaises(ValueError):
                model.AddMeanFeature().fit(invalid)

    def test_invalid_model_specifications_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            bad = [dict(settings(), model="OtherModel"),
                   {"model": "RandomForestRegressor", "parameters": {"random_state": 7, "n_jobs": 1}},
                   {"model": "RandomForestRegressor", "parameters": {"random_state": 42, "n_jobs": 4}}]
            with patch.object(model, "PARAMS_PATH", path):
                for configuration in bad:
                    path.write_text(json.dumps(configuration), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        model.build_model()

    def test_data_fingerprints_accept_line_endings_but_reject_changed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_bytes(b"1 2\n3 4\n")
            recorded = run.source_fingerprints(path)
            path.write_bytes(b"1 2\r\n3 4\r\n")
            self.assertTrue(run.matches_source(recorded, run.source_fingerprints(path)))
            path.write_bytes(b"1 2\n3 5\n")
            self.assertFalse(run.matches_source(recorded, run.source_fingerprints(path)))

    def test_scores_match_hand_calculation_and_reject_invalid_predictions(self):
        result = run.scores([1., 3., 5.], [1., 2., 6.])
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(result["r2"], .75)
        for prediction in ([1.], [1., np.nan, 2.], [[1.], [2.], [3.]]):
            with self.assertRaises(ValueError):
                run.scores([1., 3., 5.], prediction)

    def test_evaluation_and_saved_outputs_preserve_model_and_rows(self):
        split = example_split()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            path.write_text(json.dumps(settings()), encoding="utf-8")
            with patch.object(model, "PARAMS_PATH", path):
                estimator = model.build_model()
            result = run.evaluate(estimator, split, threads=1)
            with self.assertRaises(NotFittedError):
                check_is_fitted(estimator)
            residual = split.y_eval - np.asarray(result["predictions"])
            self.assertAlmostEqual(result["metrics"]["rmse"], np.sqrt(np.mean(residual**2)))
            output = Path(directory) / "result"
            run.save_results(result, output)
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["metrics"], result["metrics"])
            with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["row_id"] for row in rows], list(split.evaluation_ids))
            for name in ("prediction_vs_true.png", "residuals.png"):
                self.assertEqual((output / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_cross_validation_has_one_aligned_prediction_per_training_row(self):
        X = np.repeat(np.arange(12.)[:, None], 39, axis=1)
        y, ids = np.arange(9.) / 10, tuple(f"row:{i}" for i in range(9))
        calls, expected = [], {}
        def spy(estimator, split, *, threads):
            self.assertFalse(set(split.train_ids) & set(split.evaluation_ids))
            self.assertTrue(set(split.train_ids + split.evaluation_ids) <= set(ids))
            self.assertTrue((split.X_train[:, 0] < 9).all() and (split.X_eval[:, 0] < 9).all())
            prediction = split.y_eval + .01 * (len(calls) + 1)
            calls.append(split)
            expected.update(zip(split.evaluation_ids, prediction))
            return {"metrics": run.scores(split.y_eval, prediction),
                    "train_metrics": {"mae": 0., "rmse": 0., "r2": 1.},
                    "truth": split.y_eval.tolist(), "evaluation_ids": list(split.evaluation_ids),
                    "predictions": prediction.tolist(), "fit_seconds": 0.}
        with patch.object(run, "evaluate", side_effect=spy):
            summary, rows = run.cross_validate(object(), X[:9], y, ids, threads=1, seed=2026)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["row_id"] for row in rows}, set(ids))
        for row in rows:
            self.assertAlmostEqual(row["y_pred"], expected[row["row_id"]])


if __name__ == "__main__":
    unittest.main()
