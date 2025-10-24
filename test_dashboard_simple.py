#!/usr/bin/env python3
"""
Simple Dashboard Test - Streamlit
Tests if dashboard can load and display basic data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib
import numpy as np

st.set_page_config(
    page_title="Thermal Plant Monitor - Test",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Thermal Plant Monitoring Dashboard (Test)")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    """Load sensor data"""
    if os.path.exists("data/sensor_data.csv"):
        return pd.read_csv("data/sensor_data.csv", parse_dates=['timestamp'])
    return None

@st.cache_resource
def load_model():
    """Load anomaly detection model"""
    if os.path.exists("model/isolation_forest_trained.pkl"):
        return joblib.load("model/isolation_forest_trained.pkl")
    return None

# Load data and model
df = load_data()
model = load_model()

if df is None:
    st.error("❌ No data found. Please run: python3 examples/generate_data.py")
    st.stop()

# Plant Information
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Plant Name", df['plant_name'].iloc[0])

with col2:
    st.metric("Fuel Type", df['fuel_type'].iloc[0])

with col3:
    st.metric("Capacity", f"{df['capacity_mw'].iloc[0]:.0f} MW")

with col4:
    if model is not None:
        st.metric("Model Status", "✅ Loaded")
    else:
        st.metric("Model Status", "⚠️ Not Found")

st.markdown("---")

# Current Sensor Readings
st.subheader("📊 Current Sensor Readings")

latest = df.iloc[-1]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Temperature", f"{latest['temperature']:.1f}°C",
             delta=f"{latest['temperature'] - df['temperature'].mean():.1f}")

with col2:
    st.metric("Vibration", f"{latest['vibration']:.3f} mm/s",
             delta=f"{latest['vibration'] - df['vibration'].mean():.3f}")

with col3:
    st.metric("Pressure", f"{latest['pressure']:.1f} bar",
             delta=f"{latest['pressure'] - df['pressure'].mean():.1f}")

with col4:
    st.metric("Flow Rate", f"{latest['flow_rate']:.1f} kg/s",
             delta=f"{latest['flow_rate'] - df['flow_rate'].mean():.1f}")

with col5:
    st.metric("Load Factor", f"{latest['load_factor']:.2f}",
             delta=f"{latest['load_factor'] - df['load_factor'].mean():.2f}")

# Anomaly Detection
if model is not None:
    st.markdown("---")
    st.subheader("🔍 Anomaly Detection")

    # Run predictions
    features = ['temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
    X = df[features].values
    predictions = model.predict(X)
    scores = model.score_samples(X)

    df['anomaly'] = predictions == -1
    df['anomaly_score'] = scores

    col1, col2, col3 = st.columns(3)

    anomaly_count = (predictions == -1).sum()
    normal_count = (predictions == 1).sum()

    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Normal", f"{normal_count} ({100*normal_count/len(df):.1f}%)")
    with col3:
        st.metric("Anomalies", f"{anomaly_count} ({100*anomaly_count/len(df):.1f}%)",
                 delta=f"{anomaly_count}", delta_color="inverse")

# Time Series Plots
st.markdown("---")
st.subheader("📈 Sensor Time Series")

# Select sensor to plot
sensor_options = ['temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
selected_sensor = st.selectbox("Select Sensor", sensor_options, index=0)

# Create plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df[selected_sensor],
    mode='lines',
    name=selected_sensor.title(),
    line=dict(color='blue', width=2)
))

if model is not None:
    # Highlight anomalies
    anomaly_df = df[df['anomaly']]
    fig.add_trace(go.Scatter(
        x=anomaly_df['timestamp'],
        y=anomaly_df[selected_sensor],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))

fig.update_layout(
    title=f"{selected_sensor.title()} Over Time",
    xaxis_title="Timestamp",
    yaxis_title=selected_sensor.title(),
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# All Sensors Multi-plot
st.markdown("---")
st.subheader("📊 All Sensors Overview")

from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Temperature', 'Vibration', 'Pressure', 'Flow Rate', 'Load Factor'),
    specs=[[{}, {}, {}], [{}, {}, {"type": "table"}]]
)

sensors = ['temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

for sensor, (row, col) in zip(sensors, positions):
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df[sensor], mode='lines', name=sensor),
        row=row, col=col
    )

fig.update_layout(height=600, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Statistics Table
st.markdown("---")
st.subheader("📋 Statistical Summary")

stats_df = df[features].describe().T
stats_df['range'] = stats_df['max'] - stats_df['min']
st.dataframe(stats_df, use_container_width=True)

# Anomaly Details
if model is not None and anomaly_count > 0:
    st.markdown("---")
    st.subheader("⚠️ Detected Anomalies")

    anomaly_display = df[df['anomaly']][['timestamp'] + features + ['anomaly_score']].sort_values('anomaly_score')
    st.dataframe(anomaly_display, use_container_width=True)

    # Download button
    csv = anomaly_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Anomalies CSV",
        data=csv,
        file_name=f"anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Thermal Plant MLOps Monitoring System | Test Dashboard")
