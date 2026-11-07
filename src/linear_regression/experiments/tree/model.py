"""The final model: 39 enrichment inputs, their row mean, and a random forest."""
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, validate_data


PARAMS_PATH = Path(__file__).with_name("params.json")


class AddMeanFeature(TransformerMixin, BaseEstimator):
    """Append one within-row mean; learn no target or across-row statistic."""

    def fit(self, X, y=None):
        values = validate_data(self, X, dtype=float, ensure_all_finite=True)
        if values.shape[1] != 39:
            raise ValueError("Expected the original 39 input columns")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = validate_data(self, X, dtype=float, ensure_all_finite=True, reset=False)
        return np.column_stack((values, values.mean(axis=1)))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        names = np.asarray(input_features if input_features is not None
                           else [f"x{i}" for i in range(39)], dtype=object)
        if names.shape != (39,):
            raise ValueError("Expected 39 input feature names")
        return np.append(names, "enrichment_mean")


def load_config():
    """Read the adjacent parameter specification, without reading any dataset."""
    config = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model") != "RandomForestRegressor":
        raise ValueError("params.json must describe RandomForestRegressor")
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("params.json must contain a parameters object")
    if type(parameters.get("random_state")) is not int or parameters["random_state"] != 42:
        raise ValueError("The reproducible model requires random_state=42")
    if type(parameters.get("n_jobs")) is not int or parameters["n_jobs"] != 1:
        raise ValueError("The portable default requires n_jobs=1")
    unknown = set(parameters) - set(RandomForestRegressor().get_params())
    if unknown:
        raise ValueError(f"Unknown forest parameters: {sorted(unknown)}")
    return config


def build_model(variant="optimized"):
    """Return a fresh, unfitted final Pipeline accepting the original 39 columns."""
    if variant != "optimized":
        raise ValueError("Only the final 'optimized' model is retained")
    return Pipeline([
        ("features", AddMeanFeature()),
        ("forest", RandomForestRegressor(**load_config()["parameters"])),
    ])
