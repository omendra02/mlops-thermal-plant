#!/usr/bin/env python3
"""
Simple test script to verify basic functionality without TensorFlow
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

print("=" * 60)
print("MLOps Thermal Plant - Simple Functionality Test")
print("=" * 60)

# Test 1: Check data exists
print("\n[1/5] Checking data files...")
if os.path.exists("data/sensor_data.csv"):
    print("✓ Data file found")
    df = pd.read_csv("data/sensor_data.csv")
    print(f"✓ Loaded {len(df)} rows of sensor data")
    print(f"✓ Columns: {list(df.columns)}")
else:
    print("✗ Data file not found. Run: python3 examples/generate_data.py")
    exit(1)

# Test 2: Train simple Isolation Forest
print("\n[2/5] Training Isolation Forest model...")
features = ['temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
X = df[features].values

model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)
model.fit(X)
print("✓ Model trained successfully")

# Test 3: Make predictions
print("\n[3/5] Testing predictions...")
predictions = model.predict(X)
anomaly_count = (predictions == -1).sum()
normal_count = (predictions == 1).sum()
print(f"✓ Predictions made: {normal_count} normal, {anomaly_count} anomalies")

# Test 4: Save model
print("\n[4/5] Saving model...")
os.makedirs("model", exist_ok=True)
model_path = "model/test_isolation_forest.pkl"
joblib.dump(model, model_path)
file_size = os.path.getsize(model_path) / 1024  # KB
print(f"✓ Model saved to {model_path} ({file_size:.1f} KB)")

# Test 5: Load and verify model
print("\n[5/5] Loading and verifying model...")
loaded_model = joblib.load(model_path)
test_predictions = loaded_model.predict(X[:10])
print(f"✓ Model loaded successfully")
print(f"✓ Test predictions: {test_predictions[:5]}")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)

print("\nNext steps:")
print("1. Start dashboard: streamlit run mlops_thermal_plant/dashboard/dashboard_app.py")
print("2. Train full models: python3 scripts/train_models.py")
print("3. View MLflow: mlflow ui --port 5001")
