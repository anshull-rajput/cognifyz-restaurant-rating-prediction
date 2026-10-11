"""Predict restaurant aggregate ratings with a reproducible linear model.

Run from the repository root:
    python src/train_rating_model.py --data data/Dataset.csv

This implementation uses pandas and NumPy only. Categorical strings are
encoded with a stable feature-hashing scheme; numeric medians and scaling
statistics are learned from the training split only.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Aggregate rating"
NUMERIC_FEATURES = [
    "Country Code", "Longitude", "Latitude", "Average Cost for two",
    "Price range", "Votes",
]
CATEGORICAL_FEATURES = [
    "City", "Locality", "Cuisines", "Currency", "Has Table booking",
    "Has Online delivery", "Is delivering now", "Switch to order menu",
]
HASH_BUCKETS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def fit_encoder(training: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Learn imputers/scaling from training rows and retain category modes."""
    numeric_state = {}
    for name in numeric:
        values = pd.to_numeric(training[name], errors="coerce")
        median = float(values.median())
        filled = values.fillna(median)
        scale = float(filled.std(ddof=0)) or 1.0
        numeric_state[name] = {"median": median, "mean": float(filled.mean()), "scale": scale}

    categorical_state = {}
    for name in categorical:
        values = training[name].dropna().astype(str)
        mode = values.mode()
        categorical_state[name] = mode.iloc[0] if not mode.empty else "Unknown"
    return {"numeric": numeric_state, "categorical": categorical_state}


def transform(frame: pd.DataFrame, state: dict, numeric: list[str], categorical: list[str]) -> np.ndarray:
    """Scale numeric values and hash each categorical field into 64 indicators."""
    blocks = []
    for name in numeric:
        values = pd.to_numeric(frame[name], errors="coerce").fillna(state["numeric"][name]["median"])
        values = (values.to_numpy(dtype=float) - state["numeric"][name]["mean"]) / state["numeric"][name]["scale"]
        blocks.append(values.reshape(-1, 1))

    for name in categorical:
        mode = state["categorical"][name]
        values = frame[name].fillna(mode).astype(str).to_numpy()
        encoded = np.zeros((len(frame), HASH_BUCKETS), dtype=float)
        for row, value in enumerate(values):
            key = f"{name}\x1f{value}".encode("utf-8", errors="replace")
            encoded[row, zlib.crc32(key) % HASH_BUCKETS] = 1.0
        blocks.append(encoded)
    return np.column_stack(blocks)


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1.")

    frame = pd.read_csv(args.data, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    if TARGET not in frame:
        raise ValueError(f"Required target column {TARGET!r} is missing.")

    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=[TARGET]).copy()
    dropped_target_rows = before - len(frame)
    numeric = [name for name in NUMERIC_FEATURES if name in frame]
    categorical = [name for name in CATEGORICAL_FEATURES if name in frame]
    features = numeric + categorical
    if not features:
        raise ValueError("No expected predictor columns were found in the dataset.")

    rng = np.random.default_rng(args.random_state)
    order = rng.permutation(len(frame))
    test_count = max(1, int(np.ceil(args.test_size * len(frame))))
    test_indices, train_indices = order[:test_count], order[test_count:]
    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    encoder = fit_encoder(train, numeric, categorical)
    X_train = transform(train, encoder, numeric, categorical)
    X_test = transform(test, encoder, numeric, categorical)

    # Solve ridge regression in closed form. The intercept is not penalized.
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))
    gram = X_train.T @ X_train
    penalty = np.eye(gram.shape[0]) * args.ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, X_train.T @ y_train)
    predictions = X_test @ coefficients
    test_mse = mse(y_test, predictions)
    test_r2 = float(1.0 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

    # Permute each original input column and measure the increase in held-out
    # MSE. This ranks predictive influence without implying causation.
    baseline_mse = test_mse
    importance_rows = []
    permutation_rng = np.random.default_rng(args.random_state + 1)
    for feature in features:
        increases = []
        for _ in range(args.permutation_repeats):
            permuted = test.copy()
            permuted[feature] = permutation_rng.permutation(permuted[feature].to_numpy())
            encoded = transform(permuted, encoder, numeric, categorical)
            encoded = np.column_stack((np.ones(len(encoded)), encoded))
            increases.append(mse(y_test, encoded @ coefficients) - baseline_mse)
        importance_rows.append({
            "feature": feature,
            "mse_increase_mean": float(np.mean(increases)),
            "mse_increase_std": float(np.std(increases)),
        })
    importance = pd.DataFrame(importance_rows).sort_values(
        "mse_increase_mean", ascending=False
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    metrics = {
        "model": "Ridge regression (linear regression with L2 regularization)",
        "rows_used": int(len(frame)),
        "rows_dropped_missing_target": int(dropped_target_rows),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": features,
        "categorical_encoding": f"stable feature hashing ({HASH_BUCKETS} buckets per field)",
        "random_state": args.random_state,
        "ridge_alpha": args.ridge_alpha,
        "permutation_repeats": args.permutation_repeats,
        "mse": test_mse,
        "rmse": float(np.sqrt(test_mse)),
        "r2": test_r2,
        "target_mean_test": float(y_test.mean()),
        "target_std_test": float(y_test.std(ddof=1)),
        "top_features": importance.head(10).to_dict(orient="records"),
        "interpretation_note": "Permutation importance is predictive association, not evidence of causation.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

