"""Train and evaluate a restaurant rating regression model.

Run from the project root with:
    python src/train_rating_model.py --data data/Dataset.csv

The script uses a scikit-learn Pipeline so imputers and encoders are fitted
only on the training split. Columns that directly reveal the displayed rating
(rating label/color) and record identifiers are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Aggregate rating"
NUMERIC_FEATURES = [
    "Country Code", "Longitude", "Latitude", "Average Cost for two",
    "Price range", "Votes",
]
CATEGORICAL_FEATURES = [
    "City", "Locality", "Cuisines", "Currency", "Has Table booking",
    "Has Online delivery", "Is delivering now", "Switch to order menu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    if TARGET not in frame:
        raise ValueError(f"Required target column {TARGET!r} is missing.")

    # Rows without a known target cannot be used for supervised regression.
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=[TARGET]).copy()
    dropped_target_rows = before - len(frame)

    available_numeric = [c for c in NUMERIC_FEATURES if c in frame.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    features = available_numeric + available_categorical
    if not features:
        raise ValueError("None of the expected predictor columns were found.")

    X = frame[features].copy()
    y = frame[TARGET]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, available_numeric),
        ("categorical", categorical_pipeline, available_categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=args.trees,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.random_state,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions)

    # Sum one-hot importances back to their original input column. This is a
    # useful screening view, not a causal explanation of ratings.
    transformed_names = model.named_steps["preprocess"].get_feature_names_out()
    encoded_importance = model.named_steps["regressor"].feature_importances_
    grouped: dict[str, float] = {}
    for name, value in zip(transformed_names, encoded_importance):
        original = name.split("__", 1)[-1]
        source = next((c for c in features if original == c or original.startswith(c + "_")), original)
        grouped[source] = grouped.get(source, 0.0) + float(value)
    importance = pd.DataFrame(
        sorted(grouped.items(), key=lambda item: item[1], reverse=True),
        columns=["feature", "importance"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    metrics = {
        "model": "RandomForestRegressor",
        "rows_used": int(len(frame)),
        "rows_dropped_missing_target": int(dropped_target_rows),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": features,
        "random_state": args.random_state,
        "n_estimators": args.trees,
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "target_mean_test": float(y_test.mean()),
        "target_std_test": float(y_test.std()),
        "top_features": importance.head(10).to_dict(orient="records"),
        "interpretation_note": (
            "Feature importance describes model reliance in this split; it does not show causation."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

