"""
River water velocity estimation
Dense Optical Flow + KLT feature tracking + Regression calibration
"""

import argparse
import glob
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error


VIDEO_EXT = [
    "*.mp4", "*.avi", "*.mov", "*.mkv",
    "*.MP4", "*.AVI", "*.MOV", "*.MKV"
]


def read_meta(path):
    """
    Read metadata from the validation CSV file.

    Expected input columns:
    - Number
    - Name
    - Real (m/s)

    Returned dataframe columns used internally:
    - Name
    - Real_mps
    - time_s
    - taken_time
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}\n"
            "Please check the file path or pass it with --metadata."
        )

    # Auto-detect separator: comma, tab, or semicolon.
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize column names to the names used inside the pipeline.
    rename_map = {
        "Real (m/s)": "Real_mps",
        "Real(m/s)": "Real_mps",
        "Real": "Real_mps",
        "real": "Real_mps",
        "real_mps": "Real_mps",
        "time(s)": "time_s",
        "time (s)": "time_s",
        "time": "time_s",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["Name", "Real_mps"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

    # Convert missing labels such as '-' to NaN.
    df["Real_mps"] = df["Real_mps"].replace(["-", ""], np.nan)
    df["Real_mps"] = pd.to_numeric(df["Real_mps"], errors="coerce")

    # Optional metadata columns.
    if "time_s" not in df.columns:
        df["time_s"] = np.nan
    else:
        df["time_s"] = df["time_s"].replace(["-", ""], np.nan)
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")

    if "taken_time" not in df.columns:
        df["taken_time"] = ""

    df["Name"] = df["Name"].astype(str).str.strip()

    return df


def list_videos(video_dir):
    """List all supported video files in the selected directory."""

    out = []

    for ext in VIDEO_EXT:
        out += glob.glob(str(Path(video_dir) / ext))

    return sorted(set(out))


def find_video(video_dir, name):
    """
    Match a metadata video name to an actual video file.
    The function supports exact and relaxed name matching.
    """

    videos = list_videos(video_dir)
    target = str(name).lower()

    # Exact stem match
    for video_path in videos:
        if Path(video_path).stem.lower() == target:
            return video_path

    # Partial match
    matches = [
        video_path for video_path in videos
        if target in Path(video_path).stem.lower()
    ]

    if matches:
        return sorted(matches, key=lambda x: len(Path(x).stem))[0]

    # Relaxed match without separators
    simple_target = (
        target
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    for video_path in videos:
        stem = (
            Path(video_path).stem.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if simple_target in stem or stem in simple_target:
            return video_path

    return None


def resize(frame, max_width):
    """Resize a frame while preserving aspect ratio."""

    h, w = frame.shape[:2]

    if max_width and w > max_width:
        new_h = int(round(h * max_width / w))
        frame = cv2.resize(
            frame,
            (max_width, new_h),
            interpolation=cv2.INTER_AREA
        )

    return frame


def prep(frame, max_width):
    """Resize, normalize, enhance contrast, and denoise a frame."""

    frame = resize(frame, max_width)

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    gray = cv2.createCLAHE(
        clipLimit=2,
        tileGridSize=(8, 8)
    ).apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def load_mask(mask_path, shape):
    """Load ROI mask. If no mask exists, use the full frame."""

    h, w = shape

    if not mask_path or not Path(mask_path).exists():
        return np.ones((h, w), dtype=bool)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return np.ones((h, w), dtype=bool)

    if mask.shape != (h, w):
        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    return mask > 127


def select_rois(video_dir, meta, mask_dir, max_width=640):
    """
    Manually select the water ROI for each video.
    The selected ROI is saved as a binary mask.
    """

    Path(mask_dir).mkdir(parents=True, exist_ok=True)

    for _, row in meta.iterrows():
        name = str(row["Name"])
        video_path = find_video(video_dir, name)

        if not video_path:
            print("Video not found:", name)
            continue

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            print("Cannot read video:", name)
            continue

        frame = resize(frame, max_width)

        print("Select the water ROI, press Enter to confirm, or Esc to skip:", name)
        roi = cv2.selectROI(
            "select water ROI",
            frame,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyAllWindows()

        x, y, w, h = [int(v) for v in roi]

        if w <= 0 or h <= 0:
            print("Skipped:", name)
            continue

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255

        mask_path = Path(mask_dir) / f"{name}.png"
        cv2.imwrite(str(mask_path), mask)

        print("Mask saved:", mask_path)


def empty_klt_features():
    """Return zero-valued KLT features when tracking is not possible."""

    return {
        "klt_count": 0,
        "klt_mag_mean": 0,
        "klt_mag_med": 0,
        "klt_mag_p75": 0,
        "klt_mag_p90": 0,
        "klt_x_med": 0,
        "klt_y_med": 0,
        "klt_xy_ratio": 0,
        "klt_std": 0,
    }


def pair_features(prev, cur, mask, dt):
    """
    Extract dense optical flow and sparse KLT tracking features
    between two consecutive sampled frames.
    """

    # Dense Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        cur,
        None,
        0.5,
        3,
        25,
        3,
        5,
        1.2,
        0
    )

    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = cv2.magnitude(dx, dy)

    # Gradient is used to focus on textured/moving water patterns.
    gx = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    roi = mask.copy()

    if roi.sum() < 100:
        roi[:] = True

    if grad[roi].size > 100:
        threshold = np.percentile(grad[roi], 30)
        roi2 = roi & (grad >= threshold)

        if roi2.sum() > 100:
            roi = roi2

    m = mag[roi]
    x = dx[roi]
    y = dy[roi]

    if m.size < 100 or dt <= 0:
        return None

    # Remove extreme optical-flow values.
    lo, hi = np.percentile(m, [2, 98])
    good = (m >= lo) & (m <= hi)

    if good.sum() > 100:
        m = m[good]
        x = x[good]
        y = y[good]

    H, W = prev.shape[:2]
    diag = math.sqrt(W * W + H * H)
    eps = 1e-9

    sp = (m / dt) / diag
    sx = np.abs(x / dt) / W
    sy = np.abs(y / dt) / H

    coherence = math.sqrt(
        float(np.mean(x / (m + eps))) ** 2 +
        float(np.mean(y / (m + eps))) ** 2
    )

    out = {
        "mag_mean": np.mean(sp),
        "mag_med": np.median(sp),
        "mag_p75": np.percentile(sp, 75),
        "mag_p90": np.percentile(sp, 90),
        "mag_p95": np.percentile(sp, 95),
        "mag_std": np.std(sp),
        "x_med": np.median(sx),
        "y_med": np.median(sy),
        "xy_ratio": np.median(sx / (sx + sy + eps)),
        "coherence": coherence,
        "texture": np.mean(grad[mask]) / 255.0,
        "valid_ratio": m.size / (mask.sum() + eps),
    }

    # Sparse KLT feature tracking
    mask_u8 = (roi.astype(np.uint8) * 255)

    p0 = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        mask=mask_u8
    )

    if p0 is None or len(p0) < 10:
        out.update(empty_klt_features())
        return out

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev,
        cur,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p1 is None:
        out.update(empty_klt_features())
        return out

    good0 = p0[st.flatten() == 1].reshape(-1, 2)
    good1 = p1[st.flatten() == 1].reshape(-1, 2)

    if len(good0) < 10:
        out.update(empty_klt_features())
        return out

    disp = good1 - good0
    kdx = disp[:, 0]
    kdy = disp[:, 1]
    kmag = np.sqrt(kdx ** 2 + kdy ** 2)

    klo, khi = np.percentile(kmag, [5, 95])
    kgood = (kmag >= klo) & (kmag <= khi)

    if kgood.sum() >= 10:
        kdx = kdx[kgood]
        kdy = kdy[kgood]
        kmag = kmag[kgood]

    ksp = (kmag / dt) / diag
    ksx = np.abs(kdx / dt) / W
    ksy = np.abs(kdy / dt) / H

    out.update({
        "klt_count": len(kmag),
        "klt_mag_mean": np.mean(ksp),
        "klt_mag_med": np.median(ksp),
        "klt_mag_p75": np.percentile(ksp, 75),
        "klt_mag_p90": np.percentile(ksp, 90),
        "klt_x_med": np.median(ksx),
        "klt_y_med": np.median(ksy),
        "klt_xy_ratio": np.median(ksx / (ksx + ksy + eps)),
        "klt_std": np.std(ksp),
    })

    return out


def extract_features(
    video_path,
    mask_path=None,
    sample_fps=5,
    max_width=640,
    start_sec=0,
    end_sec=None,
    max_pairs=250
):
    """
    Extract aggregated motion features from a full video
    or from a selected time window.
    """

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or np.isnan(fps):
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else np.nan

    if end_sec is None:
        if np.isfinite(duration):
            end_sec = duration
        else:
            end_sec = start_sec + 9999

    start_frame = max(0, int(start_sec * fps))

    if frame_count:
        end_frame = min(frame_count - 1, int(end_sec * fps))
    else:
        end_frame = int(end_sec * fps)

    step = max(1, int(round(fps / sample_fps)))
    idxs = list(range(start_frame, max(start_frame + 1, end_frame), step))

    # Limit the number of frame pairs for faster processing.
    if max_pairs and len(idxs) > max_pairs + 1:
        picked = np.linspace(
            0,
            len(idxs) - 1,
            max_pairs + 1
        ).round().astype(int)

        idxs = [idxs[i] for i in sorted(set(picked))]

    prev = None
    prev_i = None
    mask = None
    rows = []
    frame_h = np.nan
    frame_w = np.nan

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        gray = prep(frame, max_width)
        frame_h, frame_w = gray.shape[:2]

        if mask is None:
            mask = load_mask(mask_path, gray.shape[:2])

        if prev is not None:
            dt = (idx - prev_i) / fps
            features = pair_features(prev, gray, mask, dt)

            if features:
                rows.append(features)

        prev = gray
        prev_i = idx

    cap.release()

    out = {
        "duration": duration,
        "width": frame_w,
        "height": frame_h,
        "pairs": len(rows),
        "aspect": frame_w / frame_h if frame_h else np.nan
    }

    if not rows:
        return out

    data = pd.DataFrame(rows)

    # Aggregate pair-level features into one feature vector.
    for col in data.columns:
        out[col + "_mean"] = data[col].mean()
        out[col + "_std"] = data[col].std(ddof=0)
        out[col + "_med"] = data[col].median()
        out[col + "_p90"] = data[col].quantile(0.90)

    return out


def models(seed=42):
    """Define regression models used for speed estimation."""

    def wrap(regressor, scale):
        steps = [("imputer", SimpleImputer(strategy="median"))]

        if scale:
            steps.append(("scaler", StandardScaler()))

        steps.append(("reg", regressor))

        return TransformedTargetRegressor(
            Pipeline(steps),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False
        )

    return {
        "Ridge": wrap(Ridge(alpha=1.0), True),

        "Huber": wrap(
            HuberRegressor(max_iter=500),
            True
        ),

        "RandomForest": wrap(
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            ),
            False
        ),

        "ExtraTrees": wrap(
            ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1
            ),
            False
        ),
    }


def metric(y, pred):
    """Compute MAE, RMSE, and MAPE."""

    mae = mean_absolute_error(y, pred)
    rmse = math.sqrt(mean_squared_error(y, pred))

    mape = np.mean(
        np.abs((y - pred) / np.maximum(np.abs(y), 1e-9))
    ) * 100

    return mae, rmse, mape


def cv_pred(model, X, y):
    """
    Run Leave-One-Out cross validation.
    Each labeled video is tested once.
    """

    pred = np.zeros(len(y))

    for train_idx, test_idx in LeaveOneOut().split(X):
        current_model = clone(model)
        current_model.fit(X.iloc[train_idx], y[train_idx])

        pred[test_idx] = np.maximum(
            current_model.predict(X.iloc[test_idx]),
            0
        )

    return pred


def add_metadata_features(df):
    """Add simple metadata-based binary features from video names."""

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


def get_feature_columns(df):
    """Select numeric feature columns used by the regression model."""

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
    ]

    return [
        col for col in df.columns
        if any(str(col).startswith(prefix) for prefix in prefixes)
    ]


def run(args):
    """
    Run video-level feature extraction, model comparison,
    evaluation, final training, and prediction.
    """

    meta = read_meta(args.metadata)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.mask_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="features"):
        name = str(row["Name"])
        video_path = find_video(args.video_dir, name)

        result_row = row.to_dict()
        result_row["video_path"] = video_path or ""
        result_row["ok"] = False

        if not video_path:
            print("Video not found:", name)
            rows.append(result_row)
            continue

        mask_path = Path(args.mask_dir) / f"{name}.png"

        try:
            features = extract_features(
                video_path,
                str(mask_path) if mask_path.exists() else None,
                sample_fps=args.sample_fps,
                max_width=args.max_width,
                start_sec=0,
                end_sec=None,
                max_pairs=args.max_pairs
            )

            result_row.update(features)
            result_row["ok"] = True

        except Exception as exc:
            print("Feature extraction error:", name, exc)

        rows.append(result_row)

    df = pd.DataFrame(rows)
    df = add_metadata_features(df)

    df.to_csv(
        Path(args.out_dir) / "features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    feature_cols = get_feature_columns(df)

    train_mask = df["Real_mps"].notna() & df["ok"].astype(bool)
    ok_mask = df["ok"].astype(bool)

    X = df.loc[train_mask, feature_cols]
    y = df.loc[train_mask, "Real_mps"].astype(float).values

    if len(y) < 4:
        raise RuntimeError(
            "Too few labeled videos were found, or the video files could not be matched."
        )

    comparisons = []
    cv_predictions = {}

    for model_name, model in models(args.seed).items():
        try:
            pred = cv_pred(model, X, y)
            mae, rmse, mape = metric(y, pred)

            comparisons.append({
                "model": model_name,
                "MAE_mps": mae,
                "RMSE_mps": rmse,
                "MAPE_percent": mape
            })

            cv_predictions[model_name] = pred

        except Exception as exc:
            comparisons.append({
                "model": model_name,
                "MAE_mps": 999,
                "RMSE_mps": 999,
                "MAPE_percent": 999,
                "error": str(exc)
            })

    comparison_df = pd.DataFrame(comparisons).sort_values("MAE_mps")

    comparison_df.to_csv(
        Path(args.out_dir) / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best_name = comparison_df.iloc[0]["model"]
    best_model = clone(models(args.seed)[best_name])
    best_model.fit(X, y)

    # Predict all videos that were successfully processed.
    df["Predicted_mps"] = np.nan
    df.loc[ok_mask, "Predicted_mps"] = np.maximum(
        best_model.predict(df.loc[ok_mask, feature_cols]),
        0
    )

    # Cross-validated prediction for labeled videos.
    df["CV_Predicted_mps"] = np.nan
    df.loc[train_mask, "CV_Predicted_mps"] = cv_predictions[best_name]

    # Use CV predictions for evaluation and final-model predictions for unlabeled videos.
    df["Eval_Predicted_mps"] = df["CV_Predicted_mps"].where(
        df["CV_Predicted_mps"].notna(),
        df["Predicted_mps"]
    )

    df["Abs_Error_mps"] = np.where(
        df["Real_mps"].notna(),
        np.abs(df["Real_mps"] - df["Eval_Predicted_mps"]),
        np.nan
    )

    df["Percent_Error"] = np.where(
        df["Real_mps"].notna(),
        df["Abs_Error_mps"] / np.maximum(np.abs(df["Real_mps"]), 1e-9) * 100,
        np.nan
    )

    for col in [
        "Predicted_mps",
        "CV_Predicted_mps",
        "Eval_Predicted_mps",
        "Abs_Error_mps",
        "Percent_Error"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    df.to_csv(
        Path(args.out_dir) / "predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "best_model": best_name
        },
        Path(args.out_dir) / "trained_model.joblib"
    )

    mae, rmse, mape = metric(y, cv_predictions[best_name])

    Path(args.out_dir, "evaluation_summary.txt").write_text(
        f"Best model: {best_name}\n"
        f"LOOCV MAE: {mae:.4f} m/s\n"
        f"LOOCV RMSE: {rmse:.4f} m/s\n"
        f"LOOCV MAPE: {mape:.2f} %\n"
        f"Use Eval_Predicted_mps for the report error table.\n",
        encoding="utf-8",
    )

    if not args.no_logs:
        make_logs(args, df, best_model, feature_cols)

    print("Done.")
    print("Best model:", best_name)
    print("Predictions table:", Path(args.out_dir) / "predictions.csv")
    print("Evaluation summary:", Path(args.out_dir) / "evaluation_summary.txt")


def make_logs(args, df, model, feature_cols):
    """Generate time-window speed logs for each video."""

    log_dir = Path(args.out_dir) / "time_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="logs"):
        if not row.get("ok", False):
            continue

        name = str(row["Name"])
        video_path = str(row["video_path"])
        duration = float(row.get("duration", np.nan))

        if not np.isfinite(duration) or duration <= 0:
            continue

        mask_path = Path(args.mask_dir) / f"{name}.png"

        records = []
        t = 0.0

        while t < duration:
            start_sec = t
            end_sec = min(duration, t + args.log_window)

            if end_sec - start_sec < max(0.35, 2 / args.sample_fps):
                break

            try:
                features = extract_features(
                    video_path,
                    str(mask_path) if mask_path.exists() else None,
                    sample_fps=args.sample_fps,
                    max_width=args.max_width,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    max_pairs=args.max_log_pairs
                )

                X_one = pd.DataFrame([features]).reindex(columns=feature_cols)
                pred = float(np.maximum(model.predict(X_one), 0)[0])

            except Exception:
                pred = np.nan

            records.append({
                "Name": name,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "predicted_speed_mps": round(pred, 4) if np.isfinite(pred) else np.nan
            })

            t += args.log_window

        pd.DataFrame(records).to_csv(
            log_dir / f"{name}_speed_log.csv",
            index=False,
            encoding="utf-8-sig"
        )


def main():
    parser = argparse.ArgumentParser(
        description="River water velocity estimation using Optical Flow, KLT, and regression models."
    )

    parser.add_argument("--video_dir", default="Dataset")
    parser.add_argument("--metadata", default="Validation Numbers.csv")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--out_dir", default="output_klt")

    parser.add_argument("--sample_fps", type=float, default=5)
    parser.add_argument("--max_width", type=int, default=640)
    parser.add_argument("--max_pairs", type=int, default=250)
    parser.add_argument("--max_log_pairs", type=int, default=20)
    parser.add_argument("--log_window", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--select_roi", action="store_true")
    parser.add_argument("--no_logs", action="store_true")

    args = parser.parse_args()

    meta = read_meta(args.metadata)

    if args.select_roi:
        select_rois(
            args.video_dir,
            meta,
            args.mask_dir,
            args.max_width
        )
        return

    run(args)


if __name__ == "__main__":
    main()