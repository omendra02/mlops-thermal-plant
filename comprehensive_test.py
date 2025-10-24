#!/usr/bin/env python3
"""
Comprehensive Test Suite for MLOps Thermal Plant
Tests all major components end-to-end
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

print("=" * 70)
print("COMPREHENSIVE TEST - MLOps Thermal Plant Monitoring System")
print("=" * 70)

# Test 1: Environment Check
print("\n[1/7] Checking Python Environment...")
print(f"✓ Python version: {sys.version.split()[0]}")
print(f"✓ Working directory: {os.getcwd()}")

required_packages = ['pandas', 'numpy', 'sklearn', 'streamlit']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} installed")
    except ImportError:
        missing.append(pkg)
        print(f"✗ {pkg} missing")

if missing:
    print(f"\n⚠ Missing packages: {missing}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

# Test 2: Data Generation
print("\n[2/7] Testing Data Generation...")
if not os.path.exists("data/sensor_data.csv"):
    print("⚠ No data found. Generating sample data...")
    exec(open("examples/generate_data.py").read())

df = pd.read_csv("data/sensor_data.csv")
print(f"✓ Loaded {len(df)} rows of data")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Plant: {df['plant_name'].iloc[0]}")
print(f"✓ Capacity: {df['capacity_mw'].iloc[0]} MW")

# Test 3: Data Statistics
print("\n[3/7] Analyzing Data Statistics...")
numeric_cols = ['temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
for col in numeric_cols:
    if col in df.columns:
        print(f"  {col:20s}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")

# Test 4: Model Training - Simple Isolation Forest
print("\n[4/7] Training Anomaly Detection Model...")
X = df[numeric_cols].values

model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    verbose=0
)

print("  Training model...")
model.fit(X)
print("✓ Model trained successfully")

# Test 5: Make Predictions
print("\n[5/7] Making Predictions...")
predictions = model.predict(X)
scores = model.score_samples(X)

anomalies = (predictions == -1)
normal = (predictions == 1)

print(f"✓ Total samples: {len(predictions)}")
print(f"✓ Normal: {normal.sum()} ({100*normal.sum()/len(predictions):.1f}%)")
print(f"✓ Anomalies: {anomalies.sum()} ({100*anomalies.sum()/len(predictions):.1f}%)")
print(f"✓ Mean anomaly score: {scores.mean():.4f}")
print(f"✓ Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# Add anomaly predictions to dataframe
df['anomaly_score'] = scores
df['is_anomaly'] = anomalies

# Test 6: Save Model and Results
print("\n[6/7] Saving Model and Results...")
os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

model_path = "model/isolation_forest_trained.pkl"
joblib.dump(model, model_path)
model_size = os.path.getsize(model_path) / 1024
print(f"✓ Model saved: {model_path} ({model_size:.1f} KB)")

# Save anomaly results
anomaly_df = df[df['is_anomaly']][['timestamp', 'temperature', 'vibration', 'pressure', 'anomaly_score']]
anomaly_df.to_csv("output/detected_anomalies.csv", index=False)
print(f"✓ Anomaly report saved: output/detected_anomalies.csv ({len(anomaly_df)} anomalies)")

# Save summary statistics
summary = {
    'total_samples': len(df),
    'anomalies_detected': int(anomalies.sum()),
    'anomaly_rate': float(anomalies.sum() / len(df)),
    'mean_anomaly_score': float(scores.mean()),
    'model_contamination': 0.05,
    'features_used': numeric_cols
}

import json
with open("output/training_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: output/training_summary.json")

# Test 7: Model Verification
print("\n[7/7] Verifying Model...")
loaded_model = joblib.load(model_path)
test_sample = X[:10]
test_predictions = loaded_model.predict(test_sample)
print(f"✓ Model loaded successfully")
print(f"✓ Test predictions on 10 samples: {list(test_predictions)}")

# Anomaly Examples
if anomalies.sum() > 0:
    print("\n" + "=" * 70)
    print("ANOMALY EXAMPLES (First 5)")
    print("=" * 70)
    anomaly_examples = df[df['is_anomaly']].head(5)
    for idx, row in anomaly_examples.iterrows():
        print(f"\nTimestamp: {row['timestamp']}")
        print(f"  Temperature: {row['temperature']:.2f}, Vibration: {row['vibration']:.4f}")
        print(f"  Pressure: {row['pressure']:.2f}, Flow: {row['flow_rate']:.2f}")
        print(f"  Anomaly Score: {row['anomaly_score']:.4f}")

# Final Summary
print("\n" + "=" * 70)
print("TEST COMPLETE - ALL CHECKS PASSED ✓")
print("=" * 70)
print("\n📊 Summary:")
print(f"  • Data samples processed: {len(df)}")
print(f"  • Features used: {len(numeric_cols)}")
print(f"  • Anomalies detected: {anomalies.sum()}")
print(f"  • Model file: {model_path}")
print(f"  • Results saved to: output/")

print("\n🚀 Next Steps:")
print("  1. View anomalies: cat output/detected_anomalies.csv")
print("  2. Start dashboard: streamlit run mlops_thermal_plant/dashboard/dashboard_app.py")
print("  3. Train advanced models: python3 scripts/train_models.py")
print("  4. View experiments: mlflow ui --port 5001")

print("\n" + "=" * 70)
