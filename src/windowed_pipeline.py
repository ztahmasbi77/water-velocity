"""
Window-based river water velocity estimation pipeline.

This script builds window-level motion features from videos, evaluates regression
models using Grouped Leave-One-Video-Out validation, and generates final
video-level predictions.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from water_velocity import (
    read_meta,
    find_video,
    extract_features,
    models,
)


def get_video_duration(video_path):
    """Return video duration in seconds using OpenCV metadata."""

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not fps or fps <= 0 or not frames or frames <= 0:
        return np.nan

    return float(frames / fps)


def add_metadata_features(df):
    """Add simple metadata features extracted from video names."""

    names = df["Name"].astype(str).str.lower()

    df["meta_time_s"] = pd.to_numeric(
        df.get("time_s", np.nan),
        errors="coerce"
    )

    df["meta_log_time_s"] = np.log1p(df["meta_time_s"])

    keywords = [
        "arrow",
        "gopro",
        "uas",
        "seeded",
        "unseeded",
        "stabilised",
        "nonstabilised",
        "notstabilised",
        "castor",
        "canada",
        "bradano",
        "alpine",
        "lamorge",
        "lavence",
        "brenta",
        "flir",
        "river",
        "noce",
        "thalhofen",
        "tiber",
        "salmon",
        "stjulien",
        "orthorectified",
    ]

    for keyword in keywords:
        df[f"meta_{keyword}"] = names.str.contains(
            keyword,
            regex=False
        ).astype(int)

    df["meta_is_short"] = (df["meta_time_s"] <= 5).astype(int)
    df["meta_is_very_short"] = (df["meta_time_s"] <= 2).astype(int)
    df["meta_is_long"] = (df["meta_time_s"] >= 50).astype(int)

    return df


def feature_columns(df):
    """Select feature columns used by the regression model."""

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
        col for col in df.columns
        if any(str(col).startswith(prefix) for prefix in prefixes)
    ]

    # Time boundaries are not used as model input features.
    drop_cols = [
        "win_start",
        "win_end",
    ]

    cols = [col for col in cols if col not in drop_cols]

    return cols


def calc_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and MAPE."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))
    ) * 100

    return mae, rmse, mape


def make_windows(duration, window, stride):
    """
    Split a video into overlapping temporal windows.

    If the video is shorter than the window length, the whole video is used
    as one window.
    """

    if not np.isfinite(duration) or duration <= 0:
        return []

    if duration <= window:
        return [(0.0, duration)]

    starts = list(
        np.arange(
            0,
            max(0, duration - window) + 1e-9,
            stride
        )
    )

    # Ensure the final part of the video is included.
    last_start = max(0.0, duration - window)

    if len(starts) == 0 or abs(starts[-1] - last_start) > 0.25:
        starts.append(last_start)

    windows = []

    for start in starts:
        end = min(duration, start + window)

        if end - start >= 0.5:
            windows.append((float(start), float(end)))

    return windows


def build_window_dataset(args):
    """
    Build a window-level dataset.

    Each video is split into time windows. For each window, motion features are
    extracted using the functions from water_velocity.py.
    """

    meta = read_meta(args.metadata)
    rows = []

    for _, row in meta.iterrows():
        name = str(row["Name"])
        video_path = find_video(args.video_dir, name)

        if not video_path:
            print("Video not found:", name)
            continue

        duration = get_video_duration(video_path)

        if not np.isfinite(duration):
            duration = float(row["time_s"]) if pd.notna(row["time_s"]) else np.nan

        windows = make_windows(duration, args.window, args.stride)

        if not windows:
            print("No valid windows:", name)
            continue

        mask_path = Path(args.mask_dir) / f"{name}.png"

        for start_sec, end_sec in windows:
            try:
                features = extract_features(
                    video_path,
                    str(mask_path) if mask_path.exists() else None,
                    sample_fps=args.sample_fps,
                    max_width=args.max_width,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    max_pairs=args.max_pairs,
                )

                out_row = row.to_dict()
                out_row.update(features)

                out_row["Name"] = name
                out_row["video_path"] = video_path
                out_row["win_start"] = start_sec
                out_row["win_end"] = end_sec
                out_row["win_len"] = end_sec - start_sec
                out_row["ok"] = True

                rows.append(out_row)

            except Exception as exc:
                print("Feature extraction error:", name, start_sec, end_sec, exc)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No valid features were extracted. "
            "Please check video paths, metadata names, masks, and video readability."
        )

    df["Real_mps"] = pd.to_numeric(df["Real_mps"], errors="coerce")
    df = add_metadata_features(df)

    return df


def video_level_prediction(preds):
    """
    Aggregate window-level predictions into one video-level prediction.

    Median is used because it is more robust to noisy windows than mean.
    """

    preds = np.asarray(preds, dtype=float)
    preds = preds[np.isfinite(preds)]

    if len(preds) == 0:
        return np.nan

    preds = np.maximum(preds, 0)

    return float(np.median(preds))


def make_sample_weights(train_df):
    """
    Give equal total weight to each video.

    If a video has N windows, each window receives weight 1/N.
    This prevents long videos from dominating the training process.
    """

    counts = train_df.groupby("Name")["Name"].transform("count").astype(float)
    weights = 1.0 / counts

    return weights.values


def fit_model_with_weights(model, X, y, train_df):
    """
    Fit the model using per-window sample weights when supported.

    The model is a TransformedTargetRegressor wrapping a Pipeline, so the
    sample weights are passed to the final regression step as reg__sample_weight.
    """

    weights = make_sample_weights(train_df)

    try:
        model.fit(X, y, reg__sample_weight=weights)
    except (TypeError, ValueError):
        model.fit(X, y)

    return model


def evaluate_models_grouped(df, fcols, seed=42):
    """
    Evaluate models using Grouped Leave-One-Video-Out validation.

    All windows from the test video are removed from training, which prevents
    data leakage between train and test.
    """

    labeled_names = sorted(
        df.loc[df["Real_mps"].notna(), "Name"].unique()
    )

    comparison = []
    all_model_preds = {}

    for model_name, model in models(seed).items():
        y_true = []
        y_pred = []
        rows = []

        for test_name in labeled_names:
            train_df = df[
                (df["Real_mps"].notna()) &
                (df["Name"] != test_name)
            ].copy()

            test_df = df[df["Name"] == test_name].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            X_train = train_df[fcols]
            y_train = train_df["Real_mps"].astype(float).values
            X_test = test_df[fcols]

            current_model = clone(model)
            current_model = fit_model_with_weights(
                current_model,
                X_train,
                y_train,
                train_df
            )

            window_preds = current_model.predict(X_test)
            pred = video_level_prediction(window_preds)

            real = float(test_df["Real_mps"].dropna().iloc[0])

            y_true.append(real)
            y_pred.append(pred)

            rows.append({
                "Name": test_name,
                "Real_mps": real,
                "Eval_Predicted_mps": pred,
            })

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae, rmse, mape = calc_metrics(y_true, y_pred)

        comparison.append({
            "model": model_name,
            "MAE_mps": mae,
            "RMSE_mps": rmse,
            "MAPE_percent": mape,
        })

        all_model_preds[model_name] = pd.DataFrame(rows)

    comparison_df = pd.DataFrame(comparison).sort_values("MAE_mps")

    return comparison_df, all_model_preds


def final_predictions(df, fcols, best_model_name, seed=42):
    """
    Train the selected model on all labeled windows and predict every video,
    including videos without ground-truth labels.
    """

    model = models(seed)[best_model_name]
    train_df = df[df["Real_mps"].notna()].copy()

    final_model = clone(model)
    final_model = fit_model_with_weights(
        final_model,
        train_df[fcols],
        train_df["Real_mps"].astype(float).values,
        train_df
    )

    rows = []

    for name in sorted(df["Name"].unique()):
        part = df[df["Name"] == name].copy()

        window_preds = final_model.predict(part[fcols])
        pred = video_level_prediction(window_preds)

        real_values = part["Real_mps"].dropna()
        real = float(real_values.iloc[0]) if len(real_values) else np.nan

        rows.append({
            "Name": name,
            "Real_mps": real,
            "Predicted_mps": pred,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Window-based river water velocity estimation pipeline."
    )

    parser.add_argument("--video_dir", default="Dataset")
    parser.add_argument("--metadata", default="Validation Numbers.csv")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--out_dir", default="outputs_windowed_pipeline")

    parser.add_argument("--sample_fps", type=float, default=10)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Building window-level features...")
    df = build_window_dataset(args)

    df.to_csv(
        Path(args.out_dir) / "window_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fcols = feature_columns(df)

    print("Number of window samples:", len(df))
    print("Number of feature columns:", len(fcols))

    print("Evaluating models with Grouped Leave-One-Video-Out...")
    comparison_df, model_preds = evaluate_models_grouped(
        df,
        fcols,
        seed=args.seed
    )

    comparison_df.to_csv(
        Path(args.out_dir) / "model_comparison_windowed.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best_model = comparison_df.iloc[0]["model"]

    print("Best model:", best_model)
    print(comparison_df)

    eval_preds = model_preds[best_model].copy()
    final_df = final_predictions(df, fcols, best_model, seed=args.seed)

    out = final_df.merge(
        eval_preds[["Name", "Eval_Predicted_mps"]],
        on="Name",
        how="left"
    )

    # Use cross-validated predictions for labeled videos and final-model
    # predictions for unlabeled videos.
    out["Eval_Predicted_mps"] = out["Eval_Predicted_mps"].where(
        out["Eval_Predicted_mps"].notna(),
        out["Predicted_mps"]
    )

    out["Abs_Error_mps"] = np.where(
        out["Real_mps"].notna(),
        np.abs(out["Real_mps"] - out["Eval_Predicted_mps"]),
        np.nan
    )

    out["Percent_Error"] = np.where(
        out["Real_mps"].notna(),
        out["Abs_Error_mps"] / np.maximum(np.abs(out["Real_mps"]), 1e-9) * 100,
        np.nan
    )

    for col in [
        "Predicted_mps",
        "Eval_Predicted_mps",
        "Abs_Error_mps",
        "Percent_Error",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)

    out.to_csv(
        Path(args.out_dir) / "predictions_windowed.csv",
        index=False,
        encoding="utf-8-sig"
    )

    labeled = out[out["Real_mps"].notna()].copy()

    mae, rmse, mape = calc_metrics(
        labeled["Real_mps"].values,
        labeled["Eval_Predicted_mps"].values
    )

    summary = (
        f"Best model: {best_model}\n"
        f"Grouped Leave-One-Video-Out Evaluation\n"
        f"MAE: {mae:.4f} m/s\n"
        f"RMSE: {rmse:.4f} m/s\n"
        f"MAPE: {mape:.2f} %\n"
        f"Window: {args.window} s\n"
        f"Stride: {args.stride} s\n"
        f"Sample FPS: {args.sample_fps}\n"
        f"Max width: {args.max_width}\n"
        f"Window samples: {len(df)}\n"
    )

    Path(args.out_dir, "evaluation_summary_windowed.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print("Saved:", Path(args.out_dir) / "predictions_windowed.csv")
    print("Saved:", Path(args.out_dir) / "model_comparison_windowed.csv")
    print("Saved:", Path(args.out_dir) / "evaluation_summary_windowed.txt")


if __name__ == "__main__":
    main()