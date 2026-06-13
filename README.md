# water-velocity
River surface velocity estimation from video using Optical Flow, KLT tracking, and machine learning.

# River Velocity Estimation from Video

This project estimates river surface flow velocity from video data using classical image processing and machine learning methods. The final pipeline combines dense Optical Flow, sparse KLT feature tracking, temporal window-based feature extraction, metadata features, and regression models.

## Method Summary

The workflow includes:

1. Reading river videos and validation metadata.
2. Selecting or loading water-region masks.
3. Extracting motion features using Optical Flow and KLT tracking.
4. Splitting videos into temporal windows.
5. Training and evaluating regression models using Grouped Leave-One-Video-Out validation.
6. Generating predictions, error analysis, plots, and window-level speed logs.

The best-performing final model was based on Random Forest regression.

## Requirements

Main Python libraries:

```bash
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

Example installation:

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib
```

## Example Commands

### 1. Create ROI masks and run video-level KLT pipeline

Select water ROI masks manually:

```bash
python water_velocity.py --select_roi
```

Run the video-level pipeline using existing masks:

```bash
python water_velocity.py
```

### 2. Run the final window-based pipeline

Simple command:

```bash
python windowed_pipeline.py --sample_fps 8 --max_width 720 --window 2 --stride 1
```

Command with explicit paths:

```bash
python src/windowed_pipeline.py --metadata "Validation Numbers.csv" --video_dir data/videos --mask_dir masks --out_dir outputs_windowed_pipeline --sample_fps 10 --max_width 960 --window 2 --stride 1
```

### 3. Generate report plots

```bash
python src/make_report_plots.py --predictions outputs_windowed_pipeline/predictions_windowed.csv --models outputs_windowed_pipeline/model_comparison_windowed.csv --out_dir report_plots
```

### 4. Run error analysis

```bash
python src/analyze_errors.py
```

### 5. Generate window-level speed logs

```bash
python make_window_speed_logs.py --window_features "outputs_windowed_pipeline/window_features.csv" --out_dir "outputs_windowed_pipeline/window_speed_logs"
```

## Main Outputs

The final pipeline generates files such as:

```text
predictions_windowed.csv
model_comparison_windowed.csv
evaluation_summary_windowed.txt
window_features.csv
window_speed_logs/
report_plots/
```

## Notes

The dataset contains heterogeneous videos with different camera angles, resolutions, lighting conditions, stabilization status, and flow patterns. Since physical calibration information such as camera height, ground sampling distance, and camera angle was not available for all videos, the method learns a regression-based mapping from visual motion features to real flow velocity.

Runtime values may vary depending on CPU performance, video resolution, sampling rate, and window size. Feature extraction is the most time-consuming part of the pipeline.
::: 
