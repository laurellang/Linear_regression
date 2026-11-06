"""Audit development data only: python -m linear_regression.experiments.data.run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from linear_regression.data import load_dataset, make_validation_split


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "dataset_UCI/raw.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs/data")
    args = parser.parse_args()
    data = load_dataset(args.input)
    split = make_validation_split(data)
    x = data.X
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    flags = (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))
    # Training thresholds are diagnostic only; no clipping or deletion is done.
    tq1, tq3 = np.quantile(split.X_train, [.25, .75], axis=0)
    summary = {
        "source": str(args.input.resolve()),
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "column_convention": "provisional: k-inf, PPPF, 39 enrichment features",
        "rows": len(data.y), "features": 39, "nonfinite_model_values": 0,
        "duplicate_model_rows": len(x) - len(np.unique(np.column_stack((x, data.y)), axis=0)),
        "duplicate_inputs": len(x) - len(np.unique(x, axis=0)),
        "constant_features": int((np.ptp(x, axis=0) == 0).sum()),
        "enrichment_min": float(x.min()), "enrichment_max": float(x.max()),
        "outside_documented_enrichment_range": int(((x < .7) | (x > 5)).sum()),
        "global_iqr_flagged_cells": int(flags.sum()),
        "training_iqr_flagged_validation_cells": int(((split.X_eval < tq1-1.5*(tq3-tq1)) | (split.X_eval > tq3+1.5*(tq3-tq1))).sum()),
        "target_quantiles": np.quantile(data.y, [0, .25, .5, .75, 1]).tolist(),
        "seed": 42, "train_rows": len(split.y_train), "validation_rows": len(split.y_eval),
        "action": "retain all rows and 39 features; no imputation, encoding or global scaling",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "feature_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "min", "q25", "median", "q75", "max", "mean", "std_population"])
        for i, col in enumerate(x.T, 1):
            writer.writerow([f"enrichment_{i:02d}", *np.quantile(col, [0, .25, .5, .75, 1]), col.mean(), col.std()])
    with (args.output / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "partition"])
        writer.writerows((i, "train") for i in split.train_ids)
        writer.writerows((i, "validation") for i in split.evaluation_ids)
    counts, edges = np.histogram(data.y, bins=30)
    bars = ''.join(f'<rect x="{60+i*20}" y="{330-c/counts.max()*260:.2f}" width="18" height="{c/counts.max()*260:.2f}" fill="#3274a1"/>' for i,c in enumerate(counts))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="740" height="400" viewBox="0 0 740 400"><rect width="740" height="400" fill="white"/><g font-family="Arial" fill="#222"><text x="60" y="32" font-size="20">Development target distribution (n={len(data.y)})</text><text x="60" y="56">Peak bin count: {counts.max()}</text>{bars}<path d="M60 65V330H660" fill="none" stroke="#222"/><text x="60" y="355">{edges[0]:.5f}</text><text x="605" y="355">{edges[-1]:.5f}</text><text x="325" y="380">k-inf</text></g></svg>'
    (args.output / "target_distribution.svg").write_text(svg, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
