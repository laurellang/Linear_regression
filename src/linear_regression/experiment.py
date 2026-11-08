"""Metrics shared by experiments that predict k-inf."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE and R2 for aligned, finite prediction vectors."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if (truth.ndim != 1 or len(truth) < 2 or prediction.shape != truth.shape
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("truth and predictions must be aligned finite vectors of length >= 2")
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }
