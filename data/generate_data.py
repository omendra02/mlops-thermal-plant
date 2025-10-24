import numpy as np
import pandas as pd
import os

# ---------------------------
# Step 1: Load & Filter Real Plant Metadata
# ---------------------------

# Ensure the data path is correct
DATA_PATH = "data/global_power_plant_database.csv"

# Check if the dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Global power plant dataset not found at {DATA_PATH}. Please provide the file.")

# Load the global power plant dataset
df_raw = pd.read_csv(DATA_PATH)

# Filter for Indian thermal plants
thermal_sources = ["Coal", "Gas", "Oil"]
df_india_thermal = df_raw[(df_raw["country"] == "IND") & (df_raw["primary_fuel"].isin(thermal_sources))]

# Check if any plants were found
if df_india_thermal.empty:
    raise ValueError("No Indian thermal plants found in the dataset. Please check the input data.")

# Save for exploration
os.makedirs("data", exist_ok=True)
df_india_thermal.to_csv("data/india_thermal_plants.csv", index=False)
print(f"Filtered {len(df_india_thermal)} Indian thermal plants.")
print("Saved to data/india_thermal_plants.csv")

# ---------------------------
# Step 2: Use Stats to Simulate Synthetic Sensor Data
# ---------------------------

# We'll simulate sensor data for just ONE plant as an example
plant = df_india_thermal.sample(1).iloc[0]
plant_name = plant['name']
capacity = plant['capacity_mw']
fuel_type = plant['primary_fuel']

print(f"Simulating data for: {plant_name} | Fuel: {fuel_type} | Capacity: {capacity} MW")

# Time series for one plant
time = pd.date_range(start="2025-01-01", periods=1000, freq="h")  # Fixed freq to 'h'

# Sensor features (simulated based on capacity, type)
temp = 70 + np.random.normal(0, 1, size=1000)
vibration = 0.3 + 0.01 * np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.02, 1000)
vibration[800:] += 0.1  # Inject fault

pressure = 180 + 0.1 * capacity + np.random.normal(0, 5, size=1000)
flow_rate = 30 + 0.05 * capacity + np.random.normal(0, 2, size=1000)
load_factor = np.clip(np.random.normal(0.85, 0.05, size=1000), 0, 1)

# Create and save DataFrame
df_synthetic = pd.DataFrame({
    "timestamp": time,
    "plant_name": plant_name,
    "fuel_type": fuel_type,
    "capacity_mw": capacity,
    "temperature": temp,
    "vibration": vibration,
    "pressure": pressure,
    "flow_rate": flow_rate,
    "load_factor": load_factor
})

df_synthetic.to_csv("data/sensor_data.csv", index=False)
print("Synthetic sensor data saved to data/sensor_data.csv")
