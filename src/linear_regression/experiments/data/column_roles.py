"""Infer column roles from the raw 41-column file and make evidence plots.

The UCI page describes 39 enrichment inputs and two simulation outputs, but the
downloaded file has no header or column-order metadata.  This module therefore
reports evidence for the project's provisional convention (columns 1 and 2 are
outputs; columns 3--41 are inputs) without presenting the inference as proof.

Run from the repository root::

    $env:PYTHONPATH = 'src'
    python -m linear_regression.experiments.data.column_roles
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ENRICHMENT_MIN = 0.7
ENRICHMENT_MAX = 5.0
EXPECTED_COLUMNS = 41


def read_raw_matrix(path: str | Path) -> np.ndarray:
    """Read and validate the headerless whitespace-separated raw matrix."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            values = line.split()
            if len(values) != EXPECTED_COLUMNS:
                raise ValueError(
                    f"{path}: line {line_number}: expected {EXPECTED_COLUMNS} "
                    f"columns, got {len(values)}"
                )
            try:
                rows.append([float(value) for value in values])
            except ValueError as error:
                raise ValueError(f"{path}: line {line_number}: nonnumeric value") from error
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: data must be finite")
    return matrix


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation, using zero for a constant input."""
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _uniform_ks_distance(values: np.ndarray) -> float:
    """Distance from Uniform(0.7, 5.0), using the two-sided empirical KS gap."""
    ordered = np.sort(values)
    expected = (ordered - ENRICHMENT_MIN) / (ENRICHMENT_MAX - ENRICHMENT_MIN)
    expected = np.clip(expected, 0.0, 1.0)
    upper = np.arange(1, len(ordered) + 1) / len(ordered)
    lower = np.arange(0, len(ordered)) / len(ordered)
    return float(max(np.max(np.abs(upper - expected)), np.max(np.abs(lower - expected))))


def summarize_columns(matrix: np.ndarray) -> list[dict[str, float | int | str]]:
    """Describe the proposed grouping, without automatically classifying roles.

    For input columns, exclude the column itself from the comparison mean to
    avoid a correlation caused simply by including a variable in its own mean.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_COLUMNS or len(matrix) < 2:
        raise ValueError(f"expected at least two rows with {EXPECTED_COLUMNS} columns")
    if not np.isfinite(matrix).all():
        raise ValueError("data must be finite")

    features = matrix[:, 2:]
    feature_mean = features.mean(axis=1)
    records: list[dict[str, float | int | str]] = []
    for index in range(EXPECTED_COLUMNS):
        values = matrix[:, index]
        if index < 2:
            role = "candidate_output"
            correlation = _correlation(values, feature_mean)
        else:
            role = "candidate_input"
            others = np.delete(features, index - 2, axis=1).mean(axis=1)
            correlation = _correlation(values, others)
        records.append(
            {
                "column": index + 1,
                "role": role,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "unique_values": int(np.unique(values).size),
                "fraction_in_enrichment_range": float(
                    np.mean((values >= ENRICHMENT_MIN) & (values <= ENRICHMENT_MAX))
                ),
                "uniform_ks_distance": _uniform_ks_distance(values),
                "corr_with_other_feature_mean": correlation,
            }
        )
    return records


def _write_evidence_plot(matrix: np.ndarray, records: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    columns = np.arange(1, EXPECTED_COLUMNS + 1)
    candidate = np.array([record["role"] == "candidate_output" for record in records])
    colors = np.where(candidate, "#d95f02", "#377eb8")
    ranges = np.array([record["range"] for record in records])
    ks = np.array([record["uniform_ks_distance"] for record in records])
    corrs = np.array([record["corr_with_other_feature_mean"] for record in records])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Evidence for column roles in raw.csv", fontsize=18, fontweight="bold", y=0.98)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d95f02", label="columns 1–2: candidate outputs"),
        Patch(facecolor="#377eb8", label="columns 3–41: candidate inputs"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))

    axes[0, 0].bar(columns, ranges, color=colors)
    axes[0, 0].set(title="Value span by column", xlabel="Column number", ylabel="max - min")
    axes[0, 0].axvline(2.5, color="#555", linestyle="--", linewidth=1)

    axes[0, 1].bar(columns, ks, color=colors)
    axes[0, 1].set(title="Distance from Uniform(0.7, 5.0)", xlabel="Column number", ylabel="KS distance")
    axes[0, 1].axvline(2.5, color="#555", linestyle="--", linewidth=1)

    axes[1, 0].bar(columns, corrs, color=colors)
    axes[1, 0].set(title="Correlation with the other-feature mean", xlabel="Column number", ylabel="Pearson r")
    axes[1, 0].axhline(0, color="#333", linewidth=0.8)
    axes[1, 0].axvline(2.5, color="#555", linestyle="--", linewidth=1)
    axes[1, 0].text(
        0.72,
        0.96,
        "col 1: r = %.3f\ncol 2: r = %.3f" % (corrs[0], corrs[1]),
        transform=axes[1, 0].transAxes,
        va="top",
        color="#d95f02",
        fontsize=10,
    )

    axes[1, 1].hist(matrix[:, 0], bins=30, density=True, alpha=0.65, color="#d95f02", label="column 1")
    axes[1, 1].hist(matrix[:, 1], bins=30, density=True, alpha=0.55, color="#e69f00", label="column 2")
    axes[1, 1].hist(matrix[:, 2:].ravel(), bins=30, density=True, alpha=0.32, color="#377eb8", label="columns 3–41 pooled")
    axes[1, 1].set(title="Distribution shape", xlabel="Value", ylabel="Density")
    axes[1, 1].legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def analyze_column_roles(input_path: str | Path, output_dir: str | Path) -> dict:
    """Write role statistics, protocol, and evidence chart; return the summary."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    matrix = read_raw_matrix(input_path)
    records = summarize_columns(matrix)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "column_role_evidence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _write_evidence_plot(matrix, records, output_dir / "column_role_evidence.png")

    summary = {
        "input": str(input_path.resolve()),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "metadata_source": "https://archive.ics.uci.edu/dataset/1091/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)",
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "provisional_convention": "columns 1-2 are simulation outputs; columns 3-41 are enrichment inputs",
        "evidence": {
            "candidate_output_ranges": [records[0]["range"], records[1]["range"]],
            "median_input_range": float(np.median([record["range"] for record in records[2:]])),
            "candidate_output_ks_distances": [records[0]["uniform_ks_distance"], records[1]["uniform_ks_distance"]],
            "median_input_ks_distance": float(np.median([record["uniform_ks_distance"] for record in records[2:]])),
            "candidate_output_correlations_with_feature_mean": [
                records[0]["corr_with_other_feature_mean"],
                records[1]["corr_with_other_feature_mean"],
            ],
            "all_inputs_in_documented_range": all(
                record["fraction_in_enrichment_range"] == 1.0 for record in records[2:]
            ),
        },
        "statistics": {
            "ks": "Two-sided empirical CDF distance from Uniform(0.7, 5.0); descriptive only, no hypothesis test",
            "correlation": "Columns 1-2 versus mean of columns 3-41; each input versus mean of the OTHER 38 inputs",
            "std": "population standard deviation (ddof=0)",
            "constant_column_correlation": "Reported as 0, not evidence of independence",
        },
        "interpretation": "The statistics support, but cannot prove, columns 1-2 as outputs. They cannot establish which is k-inf or PPPF without official column-order metadata.",
    }
    (output_dir / "column_role_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "dataset_UCI/raw.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs/data")
    args = parser.parse_args()
    summary = analyze_column_roles(args.input, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
