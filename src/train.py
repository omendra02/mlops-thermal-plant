import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# Add logging
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting model training...")

# Load data
data_path = os.path.abspath("data/sensor_data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Please generate the data first.")

df = pd.read_csv(data_path)

# Check if the data is empty
if df.empty:
    raise ValueError("The data file is empty. Please ensure the data is generated correctly.")

# Check if required columns exist
required_columns = ["temperature", "vibration"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns in the data. Expected columns: {required_columns}")

# Optional: Validate additional columns if needed
additional_columns = ["pressure", "flow_rate", "load_factor"]
if not all(col in df.columns for col in additional_columns):
    logging.warning(f"Some additional columns are missing: {additional_columns}")

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df[["temperature", "vibration"]])

# Save the model
model_path = "model/isolation_forest.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")