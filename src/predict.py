import pandas as pd
import joblib
import os
import logging

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("🔍 Starting anomaly prediction...")

# File paths
data_path = "data/sensor_data.csv"
model_path = "model/isolation_forest.pkl"
output_path = "data/sensor_data_with_anomalies.csv"

# ------------------ Load Data ------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file not found at {data_path}. Please generate the data first.")

df = pd.read_csv(data_path)

if df.empty:
    raise ValueError("❌ The data file is empty. Please ensure the data is generated correctly.")

# ------------------ Validate Columns ------------------
required_columns = ["temperature", "vibration"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns in the data: {missing_cols}")
else:
    logging.info("✅ All required columns are present in the data.")

# ------------------ Load Model ------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)
logging.info("Model loaded successfully.")

# ------------------ Predict Anomalies ------------------
try:
    # Use only the columns used during training
    df["anomaly"] = model.predict(df[["temperature", "vibration"]])
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly → 1 = anomaly
    logging.info("Anomaly prediction completed.")
except Exception as e:
    raise RuntimeError(f"❌ Error during prediction: {e}")

# ------------------ Save Output ------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
logging.info(f"📁 Predictions saved to: {output_path}")
