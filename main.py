import pandas as pd
import numpy as np
from math import log10, pi
import matplotlib.pyplot as plt
import os

# === Load & Clean the Dataset ===
try:
    df = pd.read_csv("./DataSet/3950471.csv", low_memory=False)
    print("✅ File loaded successfully.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['DATE_ONLY'] = df['DATE'].dt.date

# Filter relevant temperature data
hourly = df[['DATE', 'DATE_ONLY', 'HourlyDryBulbTemperature']].dropna()
daily_avg = df[['DATE_ONLY', 'DailyAverageDryBulbTemperature']].dropna().drop_duplicates()

# Merge on date
df_model = pd.merge(hourly, daily_avg, on='DATE_ONLY', how='inner')
df_model.rename(columns={
    'HourlyDryBulbTemperature': 'IR',
    'DailyAverageDryBulbTemperature': 'T_1day'
}, inplace=True)

# Ensure numeric
df_model['IR'] = pd.to_numeric(df_model['IR'], errors='coerce')
df_model['T_1day'] = pd.to_numeric(df_model['T_1day'], errors='coerce')
df_model = df_model.dropna(subset=['IR', 'T_1day'])

# Compute hr18
df_model['hr18'] = df_model['DATE'].dt.hour + df_model['DATE'].dt.minute / 60

# === Depth (in mm) ===
depth_mm = 76.2
log_d = log10(depth_mm)

# === Calibrated BELLS2 Model ===
df_model['T_BELLS2'] = (
    2.78 +
    0.912 * df_model['IR'] +
    (log_d - 1.25) * (
        -0.428 * df_model['IR'] +
        0.553 * df_model['T_1day'] +
        2.63 * np.sin(2 * pi * (df_model['hr18'] - 15.5) / 18)
    ) +
    0.027 * df_model['IR'] * np.sin(2 * pi * (df_model['hr18'] - 13.5) / 18)
)

# === Idaho 7-Term Model ===
beta = {
    'b0': -2.403, 'b1': 0.795, 'b2': 0.627, 'b3': 15.132,
    'b4': -2.919, 'b5': 0.062, 'b6': -2.079
}
df_model['T_Idaho'] = (
    beta['b0'] +
    beta['b1'] * df_model['IR'] +
    beta['b2'] * df_model['T_1day'] +
    beta['b3'] * log_d +
    beta['b4'] * np.sin(2 * pi * (df_model['hr18'] - 15.5) / 18) +
    beta['b5'] * df_model['IR'] * np.sin(2 * pi * (df_model['hr18'] - 13.5) / 18) +
    beta['b6'] * log_d * np.sin(2 * pi * (df_model['hr18'] - 15.5) / 18)
)

# === Save to CSV before plotting ===
try:
    output_path = os.path.join(os.getcwd(), "pavement_temp_predictions.csv")
    print("📁 Current working directory:", os.getcwd())
    print(f"📝 Attempting to save file at: {output_path}")

    df_model[['DATE', 'IR', 'T_1day', 'T_BELLS2', 'T_Idaho']].to_csv(output_path, index=False)

    if os.path.isfile(output_path):
        print(f"✅ CSV file successfully saved at: {output_path}")
    else:
        print("❌ File save operation ran, but file does NOT exist afterward.")

except Exception as e:
    print(f"❌ Error saving CSV: {e}")

# === Plot the Comparison ===
plt.figure(figsize=(14, 6))
plt.plot(df_model['DATE'], df_model['T_BELLS2'], label="Calibrated BELLS2", alpha=0.8)
plt.plot(df_model['DATE'], df_model['T_Idaho'], label="Idaho 7-Term Model", alpha=0.6, linestyle='--')
plt.title("Pavement Temperature Predictions (Depth = 76.2 mm)")
plt.xlabel("Date")
plt.ylabel("Predicted Temperature (°C)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
