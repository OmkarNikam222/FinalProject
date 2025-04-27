import pandas as pd
import numpy as np
from math import log10, pi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random

# === Step 1: Load and Prepare Data ===

df = pd.read_csv("./DataSet/3950471.csv", low_memory=False)

# Convert date
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['DATE_ONLY'] = df['DATE'].dt.date

# Filter needed columns
hourly = df[['DATE', 'DATE_ONLY', 'HourlyDryBulbTemperature']].dropna()
daily_avg = df[['DATE_ONLY', 'DailyAverageDryBulbTemperature']].dropna().drop_duplicates()

# Merge
merged = pd.merge(hourly, daily_avg, on='DATE_ONLY', how='inner')
merged.rename(columns={
    'HourlyDryBulbTemperature': 'IR',
    'DailyAverageDryBulbTemperature': 'T_1day'
}, inplace=True)

# ✅ FIX: Ensure numeric types
merged['IR'] = pd.to_numeric(merged['IR'], errors='coerce')
merged['T_1day'] = pd.to_numeric(merged['T_1day'], errors='coerce')
merged = merged.dropna(subset=['IR', 'T_1day'])

# ✅ FIX: Recompute hr18 after merge
merged['hr18'] = merged['DATE'].dt.hour + merged['DATE'].dt.minute / 60

# Depth
depth_mm = 76.2
log_d = log10(depth_mm)

# === Step 2: Generate BELLS2 Prediction ===

merged['T_BELLS2'] = (
    2.78 +
    0.912 * merged['IR'] +
    (log_d - 1.25) * (
        -0.428 * merged['IR'] +
        0.553 * merged['T_1day'] +
        2.63 * np.sin(2 * pi * (merged['hr18'] - 15.5) / 18)
    ) +
    0.027 * merged['IR'] * np.sin(2 * pi * (merged['hr18'] - 13.5) / 18)
)

# === Step 3: Prepare ML Features and Target ===

feature_cols = [
    'HourlyDryBulbTemperature',
    'HourlyDewPointTemperature',
    'HourlyWetBulbTemperature',
    'HourlyVisibility',
    'HourlyPrecipitation',
    'HourlyRelativeHumidity',
    'HourlySeaLevelPressure',
    'HourlyStationPressure'
]

# Reload original df for features
df_features = df.copy()
df_features['DATE_ONLY'] = df_features['DATE'].dt.date

# Merge again to bring in BELLS2 target
df_model = pd.merge(df_features, merged[['DATE', 'T_BELLS2']], on='DATE', how='inner')

# Drop missing
df_model = df_model.dropna(subset=feature_cols + ['T_BELLS2'])

X = df_model[feature_cols]
y = df_model['T_BELLS2']

# ✅ FIX: Ensure all feature columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# ✅ Drop rows with NaN after coercion
X = X.dropna()
y = y[X.index]

# === Step 4: Train-Test Split ===

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Train Models ===

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# === Step 6: Evaluate Models ===

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{model_name} - MAE: {mae:.2f} °C, RMSE: {rmse:.2f} °C")

print("\n📈 Model Performance Compared to BELLS2 Benchmark:")
evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# === Step 7: Improved Visualization ===

# Optional: sample smaller number of points for plotting
sample_size = 2000
if len(y_test) > sample_size:
    sampled_indices = random.sample(range(len(y_test)), sample_size)
else:
    sampled_indices = range(len(y_test))

# Scatter plot
plt.figure(figsize=(12, 6))

plt.scatter(y_test.values[sampled_indices], y_pred_lr[sampled_indices], alpha=0.6, label='Linear Regression', marker='o')
plt.scatter(y_test.values[sampled_indices], y_pred_rf[sampled_indices], alpha=0.6, label='Random Forest', marker='x')

# Line of perfect prediction
perfect = np.linspace(min(y_test), max(y_test), 100)
plt.plot(perfect, perfect, 'k--', label='Perfect Prediction')

plt.title('Predicted vs True Pavement Temperatures')
plt.xlabel('True T_BELLS2 (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
