import pandas as pd
import numpy as np
from math import log10, pi
import matplotlib.pyplot as plt

# === 1. Load and Clean Data ===
df = pd.read_csv("./DataSet/3950471.csv", low_memory=False)  # Replace with your actual filename
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['DATE_ONLY'] = df['DATE'].dt.date

# Extract relevant columns
hourly = df[['DATE', 'DATE_ONLY', 'HourlyDryBulbTemperature']].dropna()
daily_avg = df[['DATE_ONLY', 'DailyAverageDryBulbTemperature']].dropna().drop_duplicates()

# Merge to align hourly IR with daily T_1day
df_model = pd.merge(hourly, daily_avg, on='DATE_ONLY', how='inner')
df_model.rename(columns={
    'HourlyDryBulbTemperature': 'IR',
    'DailyAverageDryBulbTemperature': 'T_1day'
}, inplace=True)

# Convert temperature values to numeric
df_model['IR'] = pd.to_numeric(df_model['IR'], errors='coerce')
df_model['T_1day'] = pd.to_numeric(df_model['T_1day'], errors='coerce')
df_model = df_model.dropna(subset=['IR', 'T_1day'])

# === 2. Compute hr18 Time Component ===
df_model['hr18'] = df_model['DATE'].dt.hour + df_model['DATE'].dt.minute / 60

# === 3. Define Depth and Compute Prediction ===
depth_mm = 76.2  # e.g., 3 inches
log_depth = log10(depth_mm)

# Calibrated BELLS2 model coefficients
df_model['T_d'] = (
    2.78 +
    0.912 * df_model['IR'] +
    (log_depth - 1.25) * (
        -0.428 * df_model['IR'] +
        0.553 * df_model['T_1day'] +
        2.63 * np.sin(2 * pi * (df_model['hr18'] - 15.5) / 18)
    ) +
    0.027 * df_model['IR'] * np.sin(2 * pi * (df_model['hr18'] - 13.5) / 18)
)

# === 4. (Optional) Plot the Results ===
plt.figure(figsize=(12, 6))
plt.plot(df_model['DATE'], df_model['T_d'], label='Predicted Pavement Temp')
plt.xlabel("Timestamp")
plt.ylabel("Temperature (°C)")
plt.title("Predicted Pavement Temperature using Calibrated BELLS2")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 5. Export Cleaned Data with Predictions (Optional) ===
# df_model.to_csv("pavement_temperature_predictions.csv", index=False)
