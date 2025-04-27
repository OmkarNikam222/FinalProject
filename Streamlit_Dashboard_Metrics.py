import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import log10, pi
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide", page_title="Pavement Temp Dashboard")

# App Title
st.title("🛣️ Pavement Temperature Prediction Dashboard")

# Tabs
tab1, tab2 = st.tabs(["📂 Upload & Preview", "📈 Visualization & Metrics"])

with tab1:
    st.header("Step 1: Upload Cleaned CSV")
    uploaded_file = st.file_uploader("Upload a CSV with columns: IR, T_1day, DATE", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully uploaded!")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10))

with tab2:
    if uploaded_file:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['hr18'] = df['DATE'].dt.hour + df['DATE'].dt.minute / 60

        # Sidebar - Choose depth
        depth_mm = st.sidebar.slider("Select Depth (mm)", min_value=25, max_value=200, value=76)
        log_d = log10(depth_mm)

        # BELLS2 Calculation
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

        # Idaho 7-Term Calculation
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

        # Correct Idaho 7-Term for Alaska
        df['T_Idaho_Corrected'] = 0.7 * df['T_Idaho'] - 10

        # Random Forest Model trained on BELLS2
        feature_cols = ['IR', 'T_1day', 'hr18']
        X = df[feature_cols]
        y = df['T_BELLS2']

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        df['T_RF'] = rf_model.predict(X)

        st.subheader("📊 Interactive Plot")

        # Model selection using checkboxes
        show_bells2 = st.checkbox("Show BELLS2", value=True)
        show_idaho = st.checkbox("Show Idaho 7-Term", value=True)
        show_idaho_corr = st.checkbox("Show Idaho Corrected", value=True)
        show_rf = st.checkbox("Show Random Forest", value=True)

        fig = go.Figure()

        if show_bells2:
            fig.add_trace(go.Scatter(x=df['DATE'], y=df['T_BELLS2'], mode='lines', name='BELLS2', line=dict(width=2.5, color='#1f77b4')))
        if show_idaho:
            fig.add_trace(go.Scatter(x=df['DATE'], y=df['T_Idaho'], mode='lines', name='Idaho 7-Term', line=dict(width=2.5, color='#ff7f0e')))
        if show_idaho_corr:
            fig.add_trace(go.Scatter(x=df['DATE'], y=df['T_Idaho_Corrected'], mode='lines', name='Idaho Corrected', line=dict(width=2.5, color='#d62728')))
        if show_rf:
            fig.add_trace(go.Scatter(x=df['DATE'], y=df['T_RF'], mode='lines', name='Random Forest', line=dict(width=2.5, color='#2ca02c')))

        for y in [0, 50, 100, 150]:
            fig.add_hline(y=y, line=dict(color='gray', dash='dot', width=1), annotation_text=f"{y}°C", annotation_position="top right", opacity=0.3)

        fig.update_layout(
            title=dict(text="Pavement Temperature Predictions at {} mm Depth".format(depth_mm), font=dict(size=20), x=0.5),
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            font=dict(size=14),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=30, t=60, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📏 Model Metrics")

        mae_idaho = np.mean(np.abs(df['T_BELLS2'] - df['T_Idaho']))
        rmse_idaho = np.sqrt(np.mean((df['T_BELLS2'] - df['T_Idaho'])**2))

        mae_idaho_corr = np.mean(np.abs(df['T_BELLS2'] - df['T_Idaho_Corrected']))
        rmse_idaho_corr = np.sqrt(np.mean((df['T_BELLS2'] - df['T_Idaho_Corrected'])**2))

        mae_rf = np.mean(np.abs(df['T_BELLS2'] - df['T_RF']))
        rmse_rf = np.sqrt(np.mean((df['T_BELLS2'] - df['T_RF'])**2))

        st.metric("BELLS2 vs Idaho - MAE", f"{mae_idaho:.2f} °C")
        st.metric("BELLS2 vs Idaho - RMSE", f"{rmse_idaho:.2f} °C")

        st.metric("BELLS2 vs Idaho Corrected - MAE", f"{mae_idaho_corr:.2f} °C")
        st.metric("BELLS2 vs Idaho Corrected - RMSE", f"{rmse_idaho_corr:.2f} °C")

        st.metric("BELLS2 vs Random Forest - MAE", f"{mae_rf:.2f} °C")
        st.metric("BELLS2 vs Random Forest - RMSE", f"{rmse_rf:.2f} °C")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "pavement_predictions_updated.csv", "text/csv")

    else:
        st.warning("⚠️ Please upload a CSV in the Upload tab first.")
