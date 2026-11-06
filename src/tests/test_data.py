import tempfile
import unittest
from pathlib import Path
import numpy as np
from linear_regression.data import (RegressionDataset, load_dataset,
                                    make_validation_split, make_final_split)


class DataTests(unittest.TestCase):
    def test_parser_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("1.3\t9 " + " ".join(map(str, range(39))) + "\n")
            data = load_dataset(path)
            np.testing.assert_array_equal(data.X, [np.arange(39)])
            np.testing.assert_array_equal(data.y, [1.3])
            for text in ["", "1 2", "bad " + "0 " * 40, "nan " + "0 " * 40,
                         "1 inf " + "0 " * 39]:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "sample.csv"):
                    load_dataset(path)

    def test_split_reproducible_complete_disjoint_and_aligned(self):
        x = np.repeat(np.arange(20)[:, None], 39, axis=1)
        data = RegressionDataset(x, np.arange(20), "raw.csv")
        a, b = make_validation_split(data), make_validation_split(data)
        self.assertEqual(a.evaluation_ids, b.evaluation_ids)
        self.assertEqual(len(a.y_train), 16)
        self.assertEqual(len(a.y_eval), 4)
        self.assertFalse(set(a.train_ids) & set(a.evaluation_ids))
        self.assertEqual(len(set(a.train_ids) | set(a.evaluation_ids)), 20)
        np.testing.assert_array_equal(a.X_eval[:, 0], a.y_eval)
        self.assertNotEqual(a.evaluation_ids, make_validation_split(data, seed=7).evaluation_ids)
        for fraction in [0, 1, -1]:
            with self.assertRaises(ValueError):
                make_validation_split(data, validation_fraction=fraction)

    def test_final_overlap_and_isolation(self):
        raw = RegressionDataset(np.zeros((2,39)), np.ones(2), "raw.csv")
        same_x = RegressionDataset(np.zeros((1,39)), np.array([2]), "test.csv")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            make_final_split(raw, same_x)
        holdout = RegressionDataset(np.ones((1,39)), np.array([2]), "test.csv")
        split = make_final_split(raw, holdout)
        self.assertEqual(split.stage, "test")
        split.X_eval[0,0] = 99
        self.assertEqual(holdout.X[0,0], 1)


if __name__ == "__main__":
    unittest.main()
