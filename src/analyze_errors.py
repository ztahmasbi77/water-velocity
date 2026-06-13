import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

#df = pd.read_csv("outputs_klt/predictions.csv")
df = pd.read_csv("outputs_windowed_pipeline/predictions_windowed.csv")

df = df[df["Real_mps"].notna()].copy()
df["Abs_Error_mps"] = abs(df["Real_mps"] - df["Eval_Predicted_mps"])
df["Percent_Error"] = df["Abs_Error_mps"] / df["Real_mps"] * 100

def report(title, data):
    y = data["Real_mps"].values
    p = data["Eval_Predicted_mps"].values

    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    mape = np.mean(np.abs((y - p) / y)) * 100

    print("\n" + title)
    print("-" * 40)
    print(f"Count: {len(data)}")
    print(f"MAE:  {mae:.4f} m/s")
    print(f"RMSE: {rmse:.4f} m/s")
    print(f"MAPE: {mape:.2f} %")

report("All labeled videos", df)

report(
    "Without StJulien",
    df[df["Name"] != "StJulien"]
)

report(
    "Without top 3 high-speed difficult samples",
    df[~df["Name"].isin(["StJulien", "Tiber", "Thalhofen"])]
)

print("\nWorst errors:")
print(
    df[
        ["Name", "Real_mps", "Eval_Predicted_mps", "Abs_Error_mps", "Percent_Error"]
    ]
    .sort_values("Abs_Error_mps", ascending=False)
    .to_string(index=False)
)