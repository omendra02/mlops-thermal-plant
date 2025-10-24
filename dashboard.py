import pandas as pd
import streamlit as st
import os

# -------------------- Load Data --------------------
data_file = "data/sensor_data_with_anomalies.csv"

if not os.path.exists(data_file):
    st.error("❌ Data file not found. Please run the prediction script first.")
    st.stop()

df = pd.read_csv(data_file)

# -------------------- Metadata Header --------------------
st.set_page_config(page_title="Thermal Plant Dashboard", layout="wide")
st.title("⚙️ Thermal Plant Sensor Dashboard")

plant_name = df["plant_name"].iloc[0] if "plant_name" in df.columns else "N/A"
fuel_type = df["fuel_type"].iloc[0] if "fuel_type" in df.columns else "N/A"
capacity = df["capacity_mw"].iloc[0] if "capacity_mw" in df.columns else "N/A"

st.markdown(f"**Plant:** `{plant_name}`  |  **Fuel:** `{fuel_type}`  |  **Capacity:** `{capacity} MW`")

# -------------------- Sensor Trend Line Charts --------------------
st.subheader("📈 Sensor Trends")

sensor_cols = ["temperature", "vibration", "pressure", "flow_rate", "load_factor"]

# Line chart (interactive and separate for each)
for col in sensor_cols:
    st.line_chart(df[[col]])

# -------------------- Anomaly Insights --------------------
st.subheader("🚨 Anomalies Detected")

# Count bar chart
st.markdown("**Anomaly Count**")
st.bar_chart(df["anomaly"].value_counts())

# Data table of anomalies
st.markdown("**Anomaly Records**")
st.dataframe(df[df["anomaly"] == 1].reset_index(drop=True))

# Optional: Export anomaly table as CSV
st.download_button(
    label="📥 Download Anomaly Data",
    data=df[df["anomaly"] == 1].to_csv(index=False),
    file_name="anomalies_detected.csv",
    mime="text/csv"
)
