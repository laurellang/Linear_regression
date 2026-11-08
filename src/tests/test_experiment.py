import unittest

import numpy as np

from linear_regression.experiment import regression_metrics


class MetricTests(unittest.TestCase):
    def test_metrics_and_invalid_predictions(self):
        metrics = regression_metrics([1., 3., 5.], [1., 2., 6.])
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["r2"], .75)
        for predictions in ([1.], [1., np.nan, 2.], [[1.], [2.], [3.]]):
            with self.assertRaises(ValueError):
                regression_metrics([1., 3., 5.], predictions)


if __name__ == "__main__":
    unittest.main()
