import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10, pi

st.set_page_config(layout="wide")

# Title
st.title("Pavement Temperature Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload cleaned CSV with IR, T_1day, and DATE", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['hr18'] = df['DATE'].dt.hour + df['DATE'].dt.minute / 60

    # Sidebar: select depth
    depth_mm = st.sidebar.slider("Select Depth (mm)", min_value=25, max_value=200, value=76)
    log_d = log10(depth_mm)

    # Recalculate BELLS2 prediction
    df['T_BELLS2'] = (
        2.78 +
        0.912 * df['IR'] +
        (log_d - 1.25) * (
            -0.428 * df['IR'] +
            0.553 * df['T_1day'] +
            2.63 * np.sin(2 * pi * (df['hr18'] - 15.5) / 18)
        ) +
        0.027 * df['IR'] * np.sin(2 * pi * (df['hr18'] - 13.5) / 18)
    )

    # Recalculate Idaho model
    beta = {'b0': -2.403, 'b1': 0.795, 'b2': 0.627, 'b3': 15.132, 'b4': -2.919, 'b5': 0.062, 'b6': -2.079}
    df['T_Idaho'] = (
        beta['b0'] +
        beta['b1'] * df['IR'] +
        beta['b2'] * df['T_1day'] +
        beta['b3'] * log_d +
        beta['b4'] * np.sin(2 * pi * (df['hr18'] - 15.5) / 18) +
        beta['b5'] * df['IR'] * np.sin(2 * pi * (df['hr18'] - 13.5) / 18) +
        beta['b6'] * log_d * np.sin(2 * pi * (df['hr18'] - 15.5) / 18)
    )

    # Plot chart
    st.subheader("Predicted Temperatures vs Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['DATE'], df['T_BELLS2'], label="BELLS2", alpha=0.8)
    ax.plot(df['DATE'], df['T_Idaho'], label="Idaho 7-Term", alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Temperature Predictions at {depth_mm} mm Depth")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Metrics
    st.subheader("Model Metrics")
    mae = np.mean(np.abs(df['T_BELLS2'] - df['T_Idaho']))
    rmse = np.sqrt(np.mean((df['T_BELLS2'] - df['T_Idaho'])**2))
    st.write(f"**Mean Absolute Error (MAE)** between models: {mae:.2f} °C")
    st.write(f"**Root Mean Squared Error (RMSE)** between models: {rmse:.2f} °C")

    # Export updated CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", csv, "pavement_predictions_updated.csv", "text/csv")
