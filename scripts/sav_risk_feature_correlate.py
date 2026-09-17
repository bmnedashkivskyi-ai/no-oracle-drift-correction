"""Retrospective feature-vs-outcome correlation check, reusable across
E1-style pre-selection attempts (this is the third time this exact check
is done by hand per sam2-drift-recovery-next-directions.md SS8, SS9, SS11
-- worth having as a real script now)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy import stats  # already a transitive dep via SAM2's eval stack
import numpy as np


def load_jf_by_object(csv_path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            seq = row["sequence"].strip()
            if seq in ("", "Global score"):
                continue
            out[(seq, row["obj"].strip())] = float(row["J&F"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--capstone-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--feature-columns", nargs="+", required=True)
    args = parser.parse_args()

    capstone = load_jf_by_object(args.capstone_csv)
    baseline = load_jf_by_object(args.baseline_csv)
    deltas: dict[tuple[str, str], float] = {
        key: capstone[key] - baseline[key]
        for key in capstone if key in baseline
    }

    with args.features_csv.open() as f:
        rows = list(csv.DictReader(f))

    matched_deltas = []
    feature_values: dict[str, list[float]] = {c: [] for c in args.feature_columns}
    for row in rows:
        key = (row["video"], row["object_id"])
        if key not in deltas:
            continue
        matched_deltas.append(deltas[key])
        for col in args.feature_columns:
            feature_values[col].append(float(row[col]))

    print(f"Matched {len(matched_deltas)}/{len(rows)} objects to delta labels")
    y = np.array(matched_deltas)
    X_cols = []
    for col in args.feature_columns:
        x = np.array(feature_values[col])
        r, p = stats.pearsonr(x, y)
        print(f"{col}: pearson r={r:.3f} (p={p:.4f})")
        X_cols.append(x)

    X = np.column_stack(X_cols)
    X_design = np.column_stack([X, np.ones(len(y))])
    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"Combined R^2 ({len(args.feature_columns)} features) = {r2:.3f}")


if __name__ == "__main__":
    main()
