# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_predictions(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Name", "Real_mps"]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column not found: {c}")

    if "Eval_Predicted_mps" not in df.columns:
        if "Predicted_mps" in df.columns:
            df["Eval_Predicted_mps"] = df["Predicted_mps"]
        else:
            raise ValueError("Neither Eval_Predicted_mps nor Predicted_mps exists.")

    df["Real_mps"] = pd.to_numeric(df["Real_mps"], errors="coerce")
    df["Eval_Predicted_mps"] = pd.to_numeric(df["Eval_Predicted_mps"], errors="coerce")

    df = df[df["Real_mps"].notna() & df["Eval_Predicted_mps"].notna()].copy()

    df["Abs_Error_mps"] = np.abs(df["Real_mps"] - df["Eval_Predicted_mps"])
    df["Residual_mps"] = df["Eval_Predicted_mps"] - df["Real_mps"]
    df["Percent_Error"] = df["Abs_Error_mps"] / np.maximum(np.abs(df["Real_mps"]), 1e-9) * 100

    return df


def calc_metrics(df):
    y = df["Real_mps"].values
    p = df["Eval_Predicted_mps"].values

    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    medae = median_absolute_error(y, p)
    mape = np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-9))) * 100

    try:
        r2 = r2_score(y, p)
    except Exception:
        r2 = np.nan

    return {
        "Count": len(df),
        "MAE_mps": mae,
        "RMSE_mps": rmse,
        "MedianAE_mps": medae,
        "MAPE_percent": mape,
        "R2": r2,
    }


def save_metrics_summary(df, out_dir):
    scenarios = {
        "All labeled videos": df,
        "Without StJulien": df[df["Name"] != "StJulien"],
        "Without top 3 difficult high-speed samples": df[
            ~df["Name"].isin(["StJulien", "Tiber", "Thalhofen"])
        ],
    }

    lines = []

    for title, data in scenarios.items():
        m = calc_metrics(data)

        lines.append(title)
        lines.append("-" * 50)
        lines.append(f"Count: {m['Count']}")
        lines.append(f"MAE: {m['MAE_mps']:.4f} m/s")
        lines.append(f"RMSE: {m['RMSE_mps']:.4f} m/s")
        lines.append(f"Median AE: {m['MedianAE_mps']:.4f} m/s")
        lines.append(f"MAPE: {m['MAPE_percent']:.2f} %")
        lines.append(f"R2: {m['R2']:.4f}")
        lines.append("")

    text = "\n".join(lines)

    Path(out_dir, "metrics_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def plot_real_vs_pred(df, out_dir):
    plt.figure(figsize=(7, 7))

    x = df["Real_mps"].values
    y = df["Eval_Predicted_mps"].values

    plt.scatter(x, y, s=55, alpha=0.8)

    lim_min = 0
    lim_max = max(np.max(x), np.max(y)) * 1.08

    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.5)

    for _, row in df.iterrows():
        if row["Abs_Error_mps"] >= df["Abs_Error_mps"].quantile(0.80):
            plt.annotate(
                row["Name"],
                (row["Real_mps"], row["Eval_Predicted_mps"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points"
            )

    plt.xlabel("Real velocity (m/s)")
    plt.ylabel("Predicted velocity (m/s)")
    plt.title("Real vs predicted water velocity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "01_real_vs_predicted.png", dpi=200)
    plt.close()


def plot_abs_error_ranked(df, out_dir):
    d = df.sort_values("Abs_Error_mps", ascending=True).copy()

    plt.figure(figsize=(9, max(6, len(d) * 0.35)))
    plt.barh(d["Name"], d["Abs_Error_mps"])
    plt.xlabel("Absolute error (m/s)")
    plt.ylabel("Video")
    plt.title("Absolute error by video")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "02_absolute_error_by_video.png", dpi=200)
    plt.close()


def plot_residual_vs_real(df, out_dir):
    plt.figure(figsize=(8, 5))

    plt.scatter(df["Real_mps"], df["Residual_mps"], s=55, alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1.5)

    for _, row in df.iterrows():
        if abs(row["Residual_mps"]) >= df["Residual_mps"].abs().quantile(0.80):
            plt.annotate(
                row["Name"],
                (row["Real_mps"], row["Residual_mps"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points"
            )

    plt.xlabel("Real velocity (m/s)")
    plt.ylabel("Residual = predicted - real (m/s)")
    plt.title("Residual analysis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "03_residual_vs_real.png", dpi=200)
    plt.close()


def plot_error_distribution(df, out_dir):
    plt.figure(figsize=(8, 5))

    plt.hist(df["Abs_Error_mps"], bins=10, edgecolor="black")
    plt.xlabel("Absolute error (m/s)")
    plt.ylabel("Count")
    plt.title("Distribution of absolute errors")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "04_error_distribution.png", dpi=200)
    plt.close()


def plot_outlier_sensitivity(df, out_dir):
    scenarios = [
        ("All", df),
        ("Without StJulien", df[df["Name"] != "StJulien"]),
        (
            "Without top 3",
            df[~df["Name"].isin(["StJulien", "Tiber", "Thalhofen"])],
        ),
    ]

    rows = []

    for name, data in scenarios:
        m = calc_metrics(data)
        rows.append({
            "Scenario": name,
            "MAE": m["MAE_mps"],
            "RMSE": m["RMSE_mps"],
            "MAPE": m["MAPE_percent"],
        })

    d = pd.DataFrame(rows)

    x = np.arange(len(d))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, d["MAE"], width, label="MAE")
    plt.bar(x + width / 2, d["RMSE"], width, label="RMSE")

    plt.xticks(x, d["Scenario"])
    plt.ylabel("Error (m/s)")
    plt.title("Effect of difficult outliers on error metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "05_outlier_sensitivity_mae_rmse.png", dpi=200)
    plt.close()

    d.to_csv(Path(out_dir) / "outlier_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")


def plot_model_comparison(model_path, out_dir):
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Model comparison file not found: {model_path}")
        return

    df = pd.read_csv(model_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    if not {"model", "MAE_mps", "RMSE_mps"}.issubset(df.columns):
        print("Model comparison file does not have required columns.")
        return

    df = df.sort_values("MAE_mps").copy()

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, df["MAE_mps"], width, label="MAE")
    plt.bar(x + width / 2, df["RMSE_mps"], width, label="RMSE")

    plt.xticks(x, df["model"])
    plt.ylabel("Error (m/s)")
    plt.title("Model comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "06_model_comparison.png", dpi=200)
    plt.close()


def plot_method_improvement(out_dir):
    """
    Uses your observed results:
    - Video-level Optical Flow + metadata + RF
    - Video-level Optical Flow + KLT + metadata + RF
    - Windowed Optical Flow + KLT + metadata + RF
    """

    rows = [
        {
            "Method": "Optical Flow",
            "MAE": 0.7099,
            "RMSE": 1.3306,
            "MAPE": 93.31,
        },
        {
            "Method": "Optical Flow + KLT",
            "MAE": 0.6520,
            "RMSE": 1.2979,
            "MAPE": 83.03,
        },
        {
            "Method": "Windowed + KLT",
            "MAE": 0.4773,
            "RMSE": 1.1365,
            "MAPE": 53.06,
        },
    ]

    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df["MAE"], width, label="MAE")
    plt.bar(x + width / 2, df["RMSE"], width, label="RMSE")

    plt.xticks(x, df["Method"], rotation=10)
    plt.ylabel("Error (m/s)")
    plt.title("Improvement across development stages")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "07_method_improvement.png", dpi=200)
    plt.close()

    df.to_csv(Path(out_dir) / "method_improvement_metrics.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default="outputs_windowed_pipeline/predictions_windowed.csv",
        help="Path to predictions_windowed.csv"
    )

    parser.add_argument(
        "--models",
        default="outputs_windowed_pipeline/model_comparison_windowed.csv",
        help="Path to model_comparison_windowed.csv"
    )

    parser.add_argument(
        "--out_dir",
        default="report_plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = read_predictions(args.predictions)

    save_metrics_summary(df, args.out_dir)

    plot_real_vs_pred(df, args.out_dir)
    plot_abs_error_ranked(df, args.out_dir)
    plot_residual_vs_real(df, args.out_dir)
    plot_error_distribution(df, args.out_dir)
    plot_outlier_sensitivity(df, args.out_dir)
    plot_model_comparison(args.models, args.out_dir)
    plot_method_improvement(args.out_dir)

    print("Done. Plots saved in:", args.out_dir)


if __name__ == "__main__":
    main()