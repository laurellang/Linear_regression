"""Shared parsing and splits. Column meanings follow the project's provisional convention."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class RegressionDataset:
    X: np.ndarray
    y: np.ndarray
    source: str = "memory"

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.X.ndim != 2 or self.X.shape[1] != 39 or self.y.shape != (len(self.X),) or not len(self.X):
            raise ValueError(f"{self.source}: expected nonempty X (n, 39) and y (n,)")
        if not np.isfinite(self.X).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{self.source}: data must be finite")


@dataclass
class EvaluationSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    stage: str
    evaluation_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    seed: int | None = None
    target: str = "k-inf"


def load_dataset(path) -> RegressionDataset:
    """Read whitespace-separated 41-column data; exclude PPPF from predictors.

    Reject malformed/missing values rather than silently imputing or dropping rows.
    """
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 41:
                raise ValueError(f"{path}: line {line_number}: expected 41 columns, got {len(fields)}")
            try:
                row = [float(value) for value in fields]
            except ValueError as error:
                raise ValueError(f"{path}: line {line_number}: nonnumeric value") from error
            if not np.isfinite(row).all():
                raise ValueError(f"{path}: line {line_number}: nonfinite value")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    values = np.asarray(rows)
    return RegressionDataset(values[:, 2:], values[:, 0], str(path.resolve()))


def _ids(data, indices):
    # Physical line numbers are one-based; namespace by source file.
    return tuple(f"{Path(data.source).name}:{int(i) + 1}" for i in indices)


def make_validation_split(data, *, seed=42, validation_fraction=0.2) -> EvaluationSplit:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    n_eval = int(np.ceil(len(data.y) * validation_fraction))
    if n_eval >= len(data.y):
        raise ValueError("split must contain both training and validation samples")
    # Same shuffle convention as sklearn train_test_split for an integer seed.
    order = np.random.RandomState(seed).permutation(len(data.y))
    evaluation, training = order[:n_eval], order[n_eval:]
    return EvaluationSplit(data.X[training], data.y[training], data.X[evaluation],
                           data.y[evaluation], "validation", _ids(data, evaluation),
                           _ids(data, training), seed)


def make_final_split(raw, holdout) -> EvaluationSplit:
    """Only for final evaluation after model selection; fit nothing on holdout.

    Reject identical feature vectors even if targets differ, a stronger leakage
    check than comparing complete rows alone.
    """
    training = {tuple(row) for row in raw.X}
    if any(tuple(row) in training for row in holdout.X):
        raise ValueError(f"{raw.source} / {holdout.source}: overlapping input rows")
    return EvaluationSplit(raw.X.copy(), raw.y.copy(), holdout.X.copy(), holdout.y.copy(),
                           "test", _ids(holdout, range(len(holdout.y))),
                           _ids(raw, range(len(raw.y))))
