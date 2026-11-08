"""Compare learning curves and input-noise sensitivity on development data only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from linear_regression.data import load_dataset, make_validation_split
from linear_regression.experiment import regression_metrics
from linear_regression.experiments.linear.run import build_model as build_linear_model
from linear_regression.experiments.tree.model import build_model as build_tree_model


ROOT = Path(__file__).resolve().parents[4]
FRACTIONS = (0.1, 0.5, 1.0)
SUBSET_SEEDS = (42, 43, 44)
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.1)
NOISE_SEEDS = (101, 102, 103)
ENRICHMENT_RANGE = (0.7, 5.0)
BUILDERS = {"linear": build_linear_model, "tree": build_tree_model}


def model_spec(estimator):
    """Capture the fixed model configuration alongside each measured condition."""
    if not hasattr(estimator, "named_steps"):
        return {"class": type(estimator).__name__}
    return {name: {"class": type(step).__name__, "parameters": step.get_params(deep=False)}
            for name, step in estimator.named_steps.items()}


def sample_conditions(n_train, fractions=FRACTIONS, seeds=SUBSET_SEEDS):
    """Use nested subsets for each seed and train the full set only once."""
    if n_train < 2 or not seeds or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("expected at least two training rows, seeds and fractions in (0, 1]")
    conditions = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(n_train)
        for fraction in fractions:
            if fraction == 1.0 and seed != seeds[0]:
                continue
            count = max(2, int(round(n_train * fraction)))
            conditions.append((fraction, seed, np.sort(order[:count])))
    return conditions


def noisy_inputs(X, train_std, level, standard_normal):
    """Perturb validation inputs only; level is a fraction of training feature SD."""
    if level < 0:
        raise ValueError("noise level must be nonnegative")
    return np.asarray(X) + level * np.asarray(train_std) * standard_normal


def _ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _write_condition(directory, record, row_ids, truth, prediction):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("row_id", "y_true", "y_pred"))
        writer.writerows(zip(row_ids, truth, prediction))


def _write_summary(path, records):
    fields = ("experiment", "model", "train_fraction", "train_rows", "subset_seed",
              "noise_level", "noise_seed", "outside_range_fraction", "mae", "rmse", "r2")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record.get(key) for key in fields}, **record["metrics"]})


def _plot(records, experiment, x_key, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for model, color in (("linear", "#176d91"), ("tree", "#c26b32")):
        selected = [row for row in records if row["experiment"] == experiment and row["model"] == model]
        levels = sorted({row[x_key] for row in selected})
        mean = [np.mean([row["metrics"]["rmse"] for row in selected if row[x_key] == level])
                for level in levels]
        spread = [np.std([row["metrics"]["rmse"] for row in selected if row[x_key] == level])
                  for level in levels]
        ax.errorbar(levels, mean, yerr=spread, marker="o", capsize=3, label=model, color=color)
    ax.set(xlabel=xlabel, ylabel="Validation RMSE", title=experiment.replace("_", " ").title())
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_study(split, output_dir, builders=BUILDERS, fractions=FRACTIONS,
              subset_seeds=SUBSET_SEEDS, noise_levels=NOISE_LEVELS, noise_seeds=NOISE_SEEDS,
              data_sha256=None):
    """Fit only on split training rows; preserve the same validation rows throughout."""
    if split.stage != "validation" or not noise_seeds or 0.0 not in noise_levels:
        raise ValueError("robustness study requires a validation split and a zero-noise control")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = sample_conditions(len(split.y_train), fractions, subset_seeds)
    train_std = split.X_train.std(axis=0)
    noise = {seed: np.random.default_rng(seed).standard_normal(split.X_eval.shape)
             for seed in noise_seeds}
    records = []

    for model_name, builder in builders.items():
        full_model = None
        for fraction, seed, indices in conditions:
            estimator = builder("optimized")
            estimator.fit(split.X_train[indices], split.y_train[indices])
            prediction = estimator.predict(split.X_eval)
            metrics = regression_metrics(split.y_eval, prediction)
            record = {"experiment": "sample_size", "model": model_name,
                      "train_fraction": fraction, "train_rows": len(indices), "subset_seed": seed,
                      "stage": split.stage, "split_seed": split.seed, "target": split.target,
                      "model_spec": model_spec(estimator),
                      "train_ids": [split.train_ids[i] for i in indices],
                      "evaluation_ids": list(split.evaluation_ids), "metrics": metrics}
            records.append(record)
            label = f"fraction_{fraction:g}_seed_{seed}"
            _write_condition(output_dir / "sample_size" / model_name / label,
                             record, split.evaluation_ids, split.y_eval, prediction)
            if fraction == 1.0:
                full_model = estimator
            print(f"{model_name} sample {fraction:g} seed {seed}: RMSE {metrics['rmse']:.6f}", flush=True)

        if full_model is None:
            raise ValueError("fractions must include 1.0 for the noise experiment")
        for level in noise_levels:
            for seed in (noise_seeds[:1] if level == 0 else noise_seeds):
                perturbed = noisy_inputs(split.X_eval, train_std, level, noise[seed])
                outside = (perturbed < ENRICHMENT_RANGE[0]) | (perturbed > ENRICHMENT_RANGE[1])
                prediction = full_model.predict(perturbed)
                metrics = regression_metrics(split.y_eval, prediction)
                record = {"experiment": "input_noise", "model": model_name,
                          "train_fraction": 1.0, "train_rows": len(split.y_train),
                          "noise_level": level, "noise_seed": seed,
                          "outside_range_fraction": float(outside.mean()),
                          "stage": split.stage, "split_seed": split.seed, "target": split.target,
                          "model_spec": model_spec(full_model),
                          "evaluation_ids": list(split.evaluation_ids), "metrics": metrics}
                records.append(record)
                label = f"level_{level:g}_seed_{seed}"
                _write_condition(output_dir / "input_noise" / model_name / label,
                                 record, split.evaluation_ids, split.y_eval, prediction)
                print(f"{model_name} noise {level:g} seed {seed}: RMSE {metrics['rmse']:.6f}", flush=True)

    _write_summary(output_dir / "results.csv", records)
    _plot(records, "sample_size", "train_rows", "Training rows", output_dir / "sample_size.png")
    _plot(records, "input_noise", "noise_level", "Noise / training feature SD", output_dir / "input_noise.png")
    (output_dir / "protocol.json").write_text(json.dumps({
        "target": split.target, "stage": split.stage, "split_seed": split.seed,
        "data_sha256": data_sha256,
        "train_rows": len(split.y_train), "validation_rows": len(split.y_eval),
        "train_ids_sha256": _ids_sha256(split.train_ids),
        "evaluation_ids_sha256": _ids_sha256(split.evaluation_ids),
        "fractions": list(fractions), "subset_seeds": list(subset_seeds),
        "noise_levels": list(noise_levels), "noise_seeds": list(noise_seeds),
        "noise_definition": "Gaussian noise times training-set per-feature population SD",
        "noise_applied_to": "validation inputs only; no clipping",
        "enrichment_range": list(ENRICHMENT_RANGE),
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/robustness")
    args = parser.parse_args(argv)
    source = ROOT / "dataset_UCI/raw.csv"
    split = make_validation_split(load_dataset(source))
    run_study(split, args.output, data_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    print(f"Saved robustness results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
