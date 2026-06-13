# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor


def feature_columns(df):
    prefixes = [
        "mag_",
        "x_",
        "y_",
        "xy_",
        "klt_",
        "coherence",
        "texture",
        "valid_",
        "aspect",
        "duration",
        "width",
        "height",
        "pairs",
        "meta_",
        "win_",
    ]

    cols = [
        c for c in df.columns
        if any(str(c).startswith(p) for p in prefixes)
    ]

    # These are time/index columns, not input features
    drop_cols = [
        "win_start",
        "win_end",
    ]

    cols = [c for c in cols if c not in drop_cols]

    return cols


def make_sample_weights(train_df):
    """
    Give equal total weight to each video.
    If one video has many windows, each window receives smaller weight.
    """
    counts = train_df.groupby("Name")["Name"].transform("count").astype(float)
    weights = 1.0 / counts
    return weights.values


def build_model(seed=42):
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            )),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--window_features",
        default="outputs_windowed_pipeline/window_features.csv",
        help="Path to window_features.csv created by windowed_pipeline.py"
    )

    parser.add_argument(
        "--out_dir",
        default="outputs_windowed_pipeline/window_speed_logs",
        help="Output directory for window-level speed logs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.window_features, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Name", "Real_mps", "win_start", "win_end"]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Required column '{c}' not found in window_features.csv. "
                f"Available columns: {list(df.columns)}"
            )

    df["Real_mps"] = pd.to_numeric(df["Real_mps"], errors="coerce")
    df["win_start"] = pd.to_numeric(df["win_start"], errors="coerce")
    df["win_end"] = pd.to_numeric(df["win_end"], errors="coerce")

    fcols = feature_columns(df)

    if len(fcols) == 0:
        raise RuntimeError("No feature columns found.")

    train_df = df[df["Real_mps"].notna()].copy()

    if len(train_df) == 0:
        raise RuntimeError("No labeled windows found for training.")

    X_train = train_df[fcols]
    y_train = train_df["Real_mps"].astype(float).values
    weights = make_sample_weights(train_df)

    model = build_model(args.seed)

    try:
        model.fit(X_train, y_train, reg__sample_weight=weights)
    except Exception:
        # Fallback if sample_weight is not accepted in your sklearn version
        model.fit(X_train, y_train)

    df["predicted_speed_mps"] = np.maximum(
        model.predict(df[fcols]),
        0
    )

    df["predicted_speed_mps"] = df["predicted_speed_mps"].round(4)

    if "Real_mps" in df.columns:
        df["Abs_Error_mps"] = np.where(
            df["Real_mps"].notna(),
            np.abs(df["Real_mps"] - df["predicted_speed_mps"]),
            np.nan
        )
        df["Abs_Error_mps"] = pd.to_numeric(df["Abs_Error_mps"], errors="coerce").round(4)

    log_cols = [
        "Name",
        "win_start",
        "win_end",
        "predicted_speed_mps",
        "Real_mps",
        "Abs_Error_mps",
    ]

    log_cols = [c for c in log_cols if c in df.columns]

    all_logs = df[log_cols].sort_values(["Name", "win_start"]).copy()

    all_logs.to_csv(
        out_dir / "all_videos_window_speed_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Save one CSV per video
    for name, part in all_logs.groupby("Name"):
        safe_name = (
            str(name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )

        part.to_csv(
            out_dir / f"{safe_name}_speed_log.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("Done.")
    print("Saved all logs:", out_dir / "all_videos_window_speed_log.csv")
    print("Saved per-video logs in:", out_dir)


if __name__ == "__main__":
    main()