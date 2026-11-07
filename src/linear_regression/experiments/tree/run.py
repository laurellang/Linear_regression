"""Train and evaluate the final random forest; optionally verify its three-fold CV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from linear_regression.data import EvaluationSplit, load_dataset, make_validation_split
from .model import build_model, load_config


ROOT = Path(__file__).resolve().parents[4]


def source_fingerprints(path):
    content = Path(path).read_bytes()
    normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {"source_sha256": hashlib.sha256(content).hexdigest(),
            "source_sha256_lf": hashlib.sha256(normalized).hexdigest()}


def matches_source(recorded, actual):
    key = "source_sha256_lf" if recorded.get("source_sha256_lf") else "source_sha256"
    return bool(recorded.get(key)) and recorded[key] == actual.get(key)


def scores(truth, predictions):
    truth, predictions = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
    if (truth.ndim != 1 or len(truth) < 2 or predictions.shape != truth.shape
            or not np.isfinite(truth).all() or not np.isfinite(predictions).all()):
        raise ValueError("Targets and predictions must be aligned finite vectors of length >= 2")
    return {"mae": float(mean_absolute_error(truth, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
            "r2": float(r2_score(truth, predictions))}


def evaluate(estimator, split, *, threads=4):
    ids = tuple(split.evaluation_ids)
    if (threads < 1 or len(ids) != len(split.y_eval) or len(set(ids)) != len(ids)
            or set(ids).intersection(split.train_ids)):
        raise ValueError("Expected positive threads and aligned, unique, disjoint evaluation rows")
    model = clone(estimator).set_params(forest__n_jobs=threads)
    with threadpool_limits(limits=threads):
        started = perf_counter()
        model.fit(split.X_train, split.y_train)
        fit_seconds = perf_counter() - started
        model.set_params(forest__n_jobs=1)  # deterministic prediction aggregation
        prediction = model.predict(split.X_eval)
        train_prediction = model.predict(split.X_train)
    return {"stage": split.stage, "target": split.target, "seed": split.seed,
            "train_rows": len(split.y_train), "evaluation_ids": list(ids),
            "truth": np.asarray(split.y_eval).tolist(), "predictions": prediction.tolist(),
            "metrics": scores(split.y_eval, prediction),
            "train_metrics": scores(split.y_train, train_prediction),
            "parameters": estimator.named_steps["forest"].get_params(),
            "fit_workers": threads, "fit_seconds": fit_seconds}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_results(result, output_dir, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key not in {"truth", "predictions"}}
    payload["metadata"] = dict(metadata or {})
    _write_json(output / "metrics.json", payload)
    truth, prediction = np.asarray(result["truth"]), np.asarray(result["predictions"])
    residual = truth - prediction
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "y_true", "y_pred", "residual"])
        writer.writerows(zip(result["evaluation_ids"], truth, prediction, residual))
    with plt.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
        low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        limits = (low - .003, high + .003)
        ax.scatter(truth, prediction, s=8, alpha=.3, color="#176d91", edgecolors="none")
        ax.plot(limits, limits, "--", color="#475569", linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel="True k-inf", ylabel="Predicted k-inf",
               title="Final random forest | development validation")
        ax.set_aspect("equal", adjustable="box")
        metric = result["metrics"]
        ax.text(.04, .96, f"MAE {metric['mae']:.6f}\nRMSE {metric['rmse']:.6f}\nR² {metric['r2']:.4f}",
                transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "none"})
        ax.grid(alpha=.15)
        fig.savefig(output / "prediction_vs_true.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].scatter(prediction, residual, s=8, alpha=.3, color="#176d91", edgecolors="none")
        axes[0].axhline(0, color="#475569", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Predicted k-inf", ylabel="Residual (true - predicted)", title="Residual pattern")
        axes[1].hist(residual, bins=40, color="#176d91", alpha=.85)
        axes[1].axvline(0, color="#475569", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Residual (true - predicted)", ylabel="Samples", title="Residual distribution")
        fig.savefig(output / "residuals.png", dpi=180)
        plt.close(fig)


def cross_validate(estimator, X_train, y_train, train_ids, *, threads=4, seed=2026):
    """Verify the fixed model using training arrays only; do not select parameters."""
    X, y = np.asarray(X_train), np.asarray(y_train)
    if (X.ndim != 2 or X.shape[1] != 39 or y.shape != (len(X),) or len(y) < 6
            or len(train_ids) != len(y) or len(set(train_ids)) != len(y)):
        raise ValueError("Expected training X (n,39), y (n,) and unique aligned row identifiers")
    prediction, assignments, folds = np.empty(len(y)), np.zeros(len(y), dtype=int), []
    for fold, (training, held_out) in enumerate(KFold(3, shuffle=True, random_state=seed).split(X), 1):
        split = EvaluationSplit(X[training], y[training], X[held_out], y[held_out],
                                "cross_validation", tuple(train_ids[i] for i in held_out),
                                tuple(train_ids[i] for i in training), seed)
        result = evaluate(estimator, split, threads=threads)
        prediction[held_out], assignments[held_out] = result["predictions"], fold
        folds.append({"fold": fold, **result["metrics"], "train_rmse": result["train_metrics"]["rmse"]})
        print(f"CV fold {fold}/3: RMSE {result['metrics']['rmse']:.8f}", flush=True)
    errors = [fold["rmse"] for fold in folds]
    summary = {"seed": seed, "folds": 3, "rows": len(y), "metrics": scores(y, prediction),
               "fold_metrics": folds, "mean_fold_rmse": float(np.mean(errors)),
               "std_fold_rmse": float(np.std(errors))}
    rows = [{"row_id": row_id, "fold": int(fold), "y_true": float(truth), "y_pred": float(pred)}
            for row_id, fold, truth, pred in zip(train_ids, assignments, y, prediction)]
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tree")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cv", action="store_true", help="also verify fixed-model, training-only three-fold CV")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    config = load_config()
    source = ROOT / "dataset_UCI/raw.csv"
    fingerprints = source_fingerprints(source)
    split = make_validation_split(load_dataset(source))
    train_hash = hashlib.sha256("\n".join(split.train_ids).encode("utf-8")).hexdigest()
    if not matches_source(config["data"], fingerprints) or config["data"]["train_ids_sha256"] != train_hash:
        raise ValueError("Development data or training rows differ from the frozen model specification")
    model = build_model()
    print(f"Final model: {len(split.y_train)} train / {len(split.y_eval)} validation", flush=True)
    result = evaluate(model, split, threads=args.threads)
    save_results(result, args.output, metadata={
        **fingerprints, "train_ids_sha256": train_hash, "input_features": 39, "model_features": 40,
        "preprocessing": "append within-row mean; provisional input-column convention",
        "software": {"numpy": np.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "holdout_evaluated": False,
    })
    print(json.dumps(result["metrics"]), flush=True)
    if args.cv:
        summary, rows = cross_validate(model, split.X_train, split.y_train, split.train_ids, threads=args.threads)
        _write_json(args.output / "cv_metrics.json", summary)
        with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["row_id", "fold", "y_true", "y_pred"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved final-model results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
