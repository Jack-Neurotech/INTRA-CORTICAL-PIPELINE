
# ==========================================
# SCIENTIFIC BCI DECODER REPORT
#
# Creates a browser-based HTML report
# containing:
#
# - Dataset information
# - Neural population activity
# - Behavioral signal
# - Actual vs predicted velocity
# - Prediction error
# - Decoder metrics
# - Data dimensions
# ==========================================

from pathlib import Path
from io import BytesIO
import base64
import webbrowser

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# FIGURE TO BASE64
# ==========================================

def figure_to_base64(figure):

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(figure)

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return encoded


# ==========================================
# POPULATION ACTIVITY
# ==========================================

def create_population_figure(
    population_times,
    population_features
):

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    population_activity = np.mean(
        population_features,
        axis=1
    )

    axis.plot(
        population_times,
        population_activity
    )

    axis.set_title(
        "Neural Population Activity"
    )

    axis.set_xlabel(
        "Time (seconds)"
    )

    axis.set_ylabel(
        "Mean Spike Count / Bin"
    )

    axis.grid(
        alpha=0.25
    )

    return figure_to_base64(figure)


# ==========================================
# BEHAVIOR
# ==========================================

def create_behavior_figure(
    aligned_times,
    aligned_behavior
):

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        aligned_times,
        aligned_behavior[:, 0],
        label="X velocity"
    )

    axis.plot(
        aligned_times,
        aligned_behavior[:, 1],
        label="Y velocity"
    )

    axis.set_title(
        "Measured Finger Velocity"
    )

    axis.set_xlabel(
        "Time (seconds)"
    )

    axis.set_ylabel(
        "Velocity"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    return figure_to_base64(figure)


# ==========================================
# X PREDICTION
# ==========================================

def create_x_prediction_figure(
    y_test,
    predictions
):

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        y_test[:, 0],
        label="Measured X"
    )

    axis.plot(
        predictions[:, 0],
        label="Predicted X"
    )

    axis.set_title(
        "Finger Velocity X: Actual vs Predicted"
    )

    axis.set_xlabel(
        "Test Sample"
    )

    axis.set_ylabel(
        "Velocity"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    return figure_to_base64(figure)


# ==========================================
# Y PREDICTION
# ==========================================

def create_y_prediction_figure(
    y_test,
    predictions
):

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        y_test[:, 1],
        label="Measured Y"
    )

    axis.plot(
        predictions[:, 1],
        label="Predicted Y"
    )

    axis.set_title(
        "Finger Velocity Y: Actual vs Predicted"
    )

    axis.set_xlabel(
        "Test Sample"
    )

    axis.set_ylabel(
        "Velocity"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    return figure_to_base64(figure)


# ==========================================
# PREDICTION ERROR
# ==========================================

def create_error_figure(
    y_test,
    predictions
):

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    error = y_test - predictions

    axis.plot(
        error[:, 0],
        label="X error"
    )

    axis.plot(
        error[:, 1],
        label="Y error"
    )

    axis.axhline(
        0.0,
        linewidth=1
    )

    axis.set_title(
        "Decoder Prediction Error"
    )

    axis.set_xlabel(
        "Test Sample"
    )

    axis.set_ylabel(
        "Measured - Predicted"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    return figure_to_base64(figure)


# ==========================================
# METRIC CARD
# ==========================================

def metric_card(
    title,
    value
):

    return f"""
    <div class="metric-card">
        <div class="metric-title">
            {title}
        </div>

        <div class="metric-value">
            {value}
        </div>
    </div>
    """


# ==========================================
# GENERATE REPORT
# ==========================================

def generate_report(
    output_directory,
    dataset_path,
    number_of_neurons,
    number_of_trials,
    recording_duration,
    population_times,
    population_features,
    aligned_times,
    aligned_neural,
    aligned_behavior,
    y_test,
    predictions,
    metrics,
    training_shape,
    testing_shape
):

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        /
        "bci_decoder_report.html"
    )


    # ======================================
    # CREATE FIGURES
    # ======================================

    population_image = create_population_figure(
        population_times,
        population_features
    )

    behavior_image = create_behavior_figure(
        aligned_times,
        aligned_behavior
    )

    x_prediction_image = create_x_prediction_figure(
        y_test,
        predictions
    )

    y_prediction_image = create_y_prediction_figure(
        y_test,
        predictions
    )

    error_image = create_error_figure(
        y_test,
        predictions
    )


    # ======================================
    # METRICS
    # ======================================

    r2_x = float(metrics["r2"][0])
    r2_y = float(metrics["r2"][1])

    rmse_x = float(metrics["rmse"][0])
    rmse_y = float(metrics["rmse"][1])

    mae_x = float(metrics["mae"][0])
    mae_y = float(metrics["mae"][1])

    correlation_x = float(
        metrics["correlation_x"]
    )

    correlation_y = float(
        metrics["correlation_y"]
    )


    # ======================================
    # DATASET NAME
    # ======================================

    dataset_name = Path(
        dataset_path
    ).name


    # ======================================
    # HTML
    # ======================================

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Intracortical BCI Decoder Report
</title>

<style>

body {{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    margin: 0;
    padding: 0;

    background: #f4f6f8;
    color: #1f2933;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px;
}}

.header {{
    background: white;
    padding: 35px;
    border-radius: 14px;
    margin-bottom: 25px;

    box-shadow:
        0 2px 10px
        rgba(0, 0, 0, 0.06);
}}

h1 {{
    margin-top: 0;
    font-size: 32px;
}}

h2 {{
    margin-top: 0;
    font-size: 22px;
}}

.section {{
    background: white;
    padding: 30px;
    border-radius: 14px;
    margin-bottom: 25px;

    box-shadow:
        0 2px 10px
        rgba(0, 0, 0, 0.05);
}}

.summary-grid {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );

    gap: 15px;
}}

.metric-card {{
    background: #f7f8fa;
    border-radius: 10px;
    padding: 20px;
}}

.metric-title {{
    font-size: 13px;
    color: #667085;
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 25px;
    font-weight: 600;
}}

.info-grid {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(250px, 1fr)
        );

    gap: 10px;
}}

.info-item {{
    padding: 12px;
    background: #f7f8fa;
    border-radius: 8px;
}}

.figure {{
    width: 100%;
    margin-top: 15px;
}}

.figure img {{
    width: 100%;
    height: auto;
    border-radius: 8px;
}}

.footer {{
    text-align: center;
    color: #667085;
    padding: 20px;
}}

</style>

</head>

<body>

<div class="container">


<div class="header">

<h1>
Intracortical BCI Decoder
</h1>

<p>
Scientific decoding report generated from
the neural population and behavioral data
contained in the selected NWB recording.
</p>

<p>
<strong>Dataset:</strong>
{dataset_name}
</p>

</div>


<div class="section">

<h2>
Dataset Summary
</h2>

<div class="summary-grid">

{metric_card(
    "Neurons",
    number_of_neurons
)}

{metric_card(
    "Trials",
    number_of_trials
)}

{metric_card(
    "Recording Duration",
    f"{recording_duration:.2f} s"
)}

{metric_card(
    "Population Bins",
    len(population_times)
)}

</div>

</div>


<div class="section">

<h2>
Decoder Performance
</h2>

<div class="summary-grid">

{metric_card(
    "X R²",
    f"{r2_x:.4f}"
)}

{metric_card(
    "Y R²",
    f"{r2_y:.4f}"
)}

{metric_card(
    "X RMSE",
    f"{rmse_x:.4f}"
)}

{metric_card(
    "Y RMSE",
    f"{rmse_y:.4f}"
)}

{metric_card(
    "X MAE",
    f"{mae_x:.4f}"
)}

{metric_card(
    "Y MAE",
    f"{mae_y:.4f}"
)}

{metric_card(
    "X Pearson",
    f"{correlation_x:.4f}"
)}

{metric_card(
    "Y Pearson",
    f"{correlation_y:.4f}"
)}

</div>

</div>


<div class="section">

<h2>
Decoder Configuration
</h2>

<div class="info-grid">

<div class="info-item">
<strong>Model</strong>
<br>
Ridge Regression
</div>

<div class="info-item">
<strong>Population Bin Size</strong>
<br>
20 ms
</div>

<div class="info-item">
<strong>Target</strong>
<br>
Finger Velocity
</div>

<div class="info-item">
<strong>Training Samples</strong>
<br>
{training_shape[0]}
</div>

<div class="info-item">
<strong>Testing Samples</strong>
<br>
{testing_shape[0]}
</div>

<div class="info-item">
<strong>Neural Features</strong>
<br>
{training_shape[1]}
</div>

</div>

</div>


<div class="section">

<h2>
Neural Population Activity
</h2>

<p>
Mean spike count across the recorded
neural population for each 20 ms bin.
</p>

<div class="figure">

<img
src="data:image/png;base64,{population_image}"
alt="Neural population activity"
>

</div>

</div>


<div class="section">

<h2>
Measured Finger Velocity
</h2>

<div class="figure">

<img
src="data:image/png;base64,{behavior_image}"
alt="Measured finger velocity"
>

</div>

</div>


<div class="section">

<h2>
Finger Velocity X
</h2>

<div class="figure">

<img
src="data:image/png;base64,{x_prediction_image}"
alt="Actual versus predicted X velocity"
>

</div>

</div>


<div class="section">

<h2>
Finger Velocity Y
</h2>

<div class="figure">

<img
src="data:image/png;base64,{y_prediction_image}"
alt="Actual versus predicted Y velocity"
>

</div>

</div>


<div class="section">

<h2>
Decoder Prediction Error
</h2>

<div class="figure">

<img
src="data:image/png;base64,{error_image}"
alt="Decoder prediction error"
>

</div>

</div>


<div class="section">

<h2>
Data Dimensions
</h2>

<div class="info-grid">

<div class="info-item">
<strong>Population Features</strong>
<br>
{population_features.shape}
</div>

<div class="info-item">
<strong>Aligned Neural Data</strong>
<br>
{aligned_neural.shape}
</div>

<div class="info-item">
<strong>Aligned Behavior</strong>
<br>
{aligned_behavior.shape}
</div>

<div class="info-item">
<strong>Predictions</strong>
<br>
{predictions.shape}
</div>

<div class="info-item">
<strong>Training Set</strong>
<br>
{training_shape}
</div>

<div class="info-item">
<strong>Testing Set</strong>
<br>
{testing_shape}
</div>

</div>

</div>


<div class="footer">

Intracortical BCI Decoder

<br>

Neural population → behavioral decoding

</div>


</div>

</body>

</html>
"""


    # ======================================
    # WRITE REPORT
    # ======================================

    output_file.write_text(
        html,
        encoding="utf-8"
    )


    # ======================================
    # OPEN REPORT
    # ======================================

    webbrowser.open(
        output_file.resolve().as_uri()
    )

    return output_file.resolve()
