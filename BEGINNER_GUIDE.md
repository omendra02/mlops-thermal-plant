# 🌟 Thermal Plant MLOps - Complete Beginner's Guide

## 🎯 **What is this project?**

Imagine you have a **thermal power plant** (like a big factory that makes electricity from coal). This plant has many **sensors** that measure things like:
- 🌡️ Temperature (how hot the steam is)
- ⚙️ Vibration (how much the machines are shaking)
- 🔧 Pressure (how much force the steam has)
- 💧 Flow rate (how fast water/steam is moving)
- ⚡ Load factor (how much electricity is being made)

**The Problem:** Sometimes these sensors show weird readings that could mean something is broken or dangerous. We need to **automatically detect these problems** before they cause big issues.

**Our Solution:** We built an **AI system** that:
1. 📊 Collects sensor data
2. 🤖 Learns what "normal" looks like
3. 🚨 Detects when something is "abnormal" (anomalies)
4. 📱 Shows everything on a dashboard

---

## 🏗️ **Project Structure (What Files Do What)**

```
mlops-thermal-plant/
├── 📊 data/                    # Where we store information
│   ├── generate_data.py       # Creates fake sensor data (for testing)
│   ├── sensor_data.csv        # The sensor readings we collected
│   └── sensor_data_with_anomalies.csv  # Data + AI predictions
├── 🤖 src/                     # The "brain" of our system
│   ├── train.py              # Teaches the AI what's normal
│   └── predict.py            # Uses AI to find problems
├── 💾 model/                   # Where we save the trained AI
│   └── isolation_forest.pkl  # The trained AI brain
├── 📱 dashboard.py            # The website interface
├── 🔧 config/                 # Settings and configurations
├── 🐳 Docker files           # For easy deployment
└── 📚 Documentation          # Help files
```

---

## 🔄 **How Everything Works (Step by Step)**

### **Step 1: Generate Data** 📊
```python
# data/generate_data.py
```
**What it does:**
- Creates fake sensor data that looks like real power plant data
- Uses real Indian power plant information (ZAWAR MINES, 80MW Coal plant)
- Generates 1000+ sensor readings with realistic values

**Why we need this:**
- Real power plants don't let us connect to their sensors easily
- We need data to train our AI
- This creates realistic test data

**Example data it creates:**
```
Timestamp: 2025-01-01 00:00:00
Temperature: 485.2°C
Vibration: 1.1 mm/s
Pressure: 152.3 bar
Flow Rate: 62.1 kg/s
Load Factor: 85.3%
```

### **Step 2: Train the AI** 🤖
```python
# src/train.py
```
**What it does:**
- Takes all the sensor data
- Teaches an AI algorithm called "Isolation Forest" what normal looks like
- Saves the trained AI to a file

**How it works:**
1. Load sensor data from CSV file
2. Check if data has required columns (temperature, vibration)
3. Train Isolation Forest algorithm
4. Save trained model to `model/isolation_forest.pkl`

**Isolation Forest explained:**
- It's like teaching a detective to spot suspicious behavior
- Normal sensor readings = normal behavior
- Unusual readings = suspicious behavior
- The AI learns patterns and can spot outliers

### **Step 3: Detect Anomalies** 🔍
```python
# src/predict.py
```
**What it does:**
- Loads the trained AI model
- Runs it on all sensor data
- Identifies which readings are "anomalous" (weird/problematic)
- Saves results with anomaly flags

**How it works:**
1. Load sensor data
2. Load trained AI model
3. Run AI on each sensor reading
4. Mark readings as "normal" (0) or "anomaly" (1)
5. Save results to new CSV file

**Example output:**
```
Temperature: 485.2°C → Normal (0)
Temperature: 650.1°C → Anomaly (1) ⚠️
Vibration: 1.1 mm/s → Normal (0)
Vibration: 8.5 mm/s → Anomaly (1) ⚠️
```

### **Step 4: Show Dashboard** 📱
```python
# dashboard.py
```
**What it does:**
- Creates a website interface using Streamlit
- Shows sensor data as interactive charts
- Displays detected anomalies
- Allows downloading of results

**What you see:**
- Plant information (name, fuel type, capacity)
- Line charts showing sensor trends over time
- Bar chart showing how many anomalies were found
- Table of all anomaly records
- Download button for anomaly data

---

## 🔄 **Complete Data Flow Diagram**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🌡️ SENSORS    │───▶│  📊 DATA GEN    │───▶│  🤖 AI TRAIN    │───▶│  🔍 PREDICT     │
│                 │    │                 │    │                 │    │                 │
│ Temperature     │    │ sensor_data.csv │    │ isolation_forest│    │ anomalies.csv   │
│ Vibration       │    │ (1000+ records) │    │ .pkl            │    │ (50 anomalies)  │
│ Pressure        │    │                 │    │                 │    │                 │
│ Flow Rate       │    │                 │    │                 │    │                 │
│ Load Factor     │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  📱 DASHBOARD   │
                                                │                 │
                                                │ Interactive     │
                                                │ Charts & Tables │
                                                │ Anomaly Alerts  │
                                                │ Data Export     │
                                                └─────────────────┘
```

---

## 📊 **Detailed Code Explanation**

### **1. Data Generation (`data/generate_data.py`)**

```python
# This creates realistic power plant sensor data
import numpy as np
import pandas as pd

# Create 1000 time points (one per hour for ~40 days)
N = 1000
timestamps = pd.date_range(start="2025-01-01", periods=N, freq="H")

# Generate realistic sensor values with some randomness
steam_temp = 500 + np.random.normal(0, 10, size=N)  # Around 500°C ± 10°
vibration = 0.3 + 0.01 * np.sin(...) + np.random.normal(0, 0.02, N)  # Smooth + noise
vibration[800:] += 0.1  # Inject some anomalies after hour 800

# Save to CSV file
df.to_csv("data/sensor_data.csv")
```

**What this does:**
- Creates 1000 hours of fake sensor data
- Uses realistic ranges for thermal power plant sensors
- Adds some "faults" (anomalies) that our AI should detect
- Saves everything to a CSV file

### **2. Model Training (`src/train.py`)**

```python
# This teaches the AI what normal sensor readings look like
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the sensor data
df = pd.read_csv("data/sensor_data.csv")

# Check we have the right data
required_columns = ["temperature", "vibration"]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")

# Train the AI (Isolation Forest algorithm)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df[["temperature", "vibration"]])

# Save the trained AI
joblib.dump(model, "model/isolation_forest.pkl")
```

**What this does:**
- Loads sensor data from CSV
- Validates that we have temperature and vibration data
- Trains Isolation Forest algorithm with 5% contamination (expects 5% anomalies)
- Saves the trained model so we can use it later

### **3. Anomaly Prediction (`src/predict.py`)**

```python
# This uses the trained AI to find anomalies in the data
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv("data/sensor_data.csv")
model = joblib.load("model/isolation_forest.pkl")

# Run AI on all data points
df["anomaly"] = model.predict(df[["temperature", "vibration"]])

# Convert from -1/1 to 0/1 (0=normal, 1=anomaly)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

# Save results
df.to_csv("data/sensor_data_with_anomalies.csv")
```

**What this does:**
- Loads the sensor data and trained AI model
- Runs the AI on every sensor reading
- Converts AI output (-1/1) to human-readable format (0/1)
- Saves results with anomaly flags

### **4. Dashboard (`dashboard.py`)**

```python
# This creates a website to show everything
import streamlit as st
import pandas as pd

# Load the data with anomalies
df = pd.read_csv("data/sensor_data_with_anomalies.csv")

# Create the website
st.title("⚙️ Thermal Plant Sensor Dashboard")

# Show plant info
plant_name = df["plant_name"].iloc[0]
st.markdown(f"**Plant:** {plant_name}")

# Show sensor charts
for col in ["temperature", "vibration", "pressure"]:
    st.line_chart(df[[col]])

# Show anomalies
st.subheader("🚨 Anomalies Detected")
st.bar_chart(df["anomaly"].value_counts())
st.dataframe(df[df["anomaly"] == 1])
```

**What this does:**
- Creates a web interface using Streamlit
- Shows plant information at the top
- Creates line charts for each sensor type
- Shows how many anomalies were found
- Displays a table of all anomaly records

---

## 🎯 **Real-World Example**

Let's say you're monitoring a thermal power plant:

### **Normal Day:**
```
🌡️ Steam Temperature: 485°C (Normal)
⚙️ Vibration: 1.1 mm/s (Normal)  
🔧 Pressure: 152 bar (Normal)
💧 Flow Rate: 62 kg/s (Normal)
⚡ Load Factor: 85% (Normal)
```

### **Problem Day:**
```
🌡️ Steam Temperature: 485°C (Normal)
⚙️ Vibration: 1.1 mm/s (Normal)
🔧 Pressure: 152 bar (Normal)
💧 Flow Rate: 62 kg/s (Normal)
⚡ Load Factor: 85% (Normal)

--- 2 hours later ---

🌡️ Steam Temperature: 485°C (Normal)
⚙️ Vibration: 8.5 mm/s (ANOMALY! 🚨)  ← Machine is shaking too much!
🔧 Pressure: 152 bar (Normal)
💧 Flow Rate: 62 kg/s (Normal)
⚡ Load Factor: 85% (Normal)
```

**Our AI would detect:** "Hey! The vibration is way higher than normal. Something might be wrong with the turbine!"

---

## 🚀 **How to Run Everything**

### **Step 1: Setup**
```bash
cd mlops-thermal-plant
source .venv/bin/activate  # Activate Python environment
```

### **Step 2: Generate Data**
```bash
python data/generate_data.py
# Creates: data/sensor_data.csv (1000+ sensor readings)
```

### **Step 3: Train AI**
```bash
python src/train.py
# Creates: model/isolation_forest.pkl (trained AI brain)
```

### **Step 4: Find Anomalies**
```bash
python src/predict.py
# Creates: data/sensor_data_with_anomalies.csv (data + anomaly flags)
```

### **Step 5: View Dashboard**
```bash
streamlit run dashboard.py
# Opens: http://localhost:8501 (interactive website)
```

---

## 📊 **What You'll See in the Dashboard**

### **Dashboard Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ Thermal Plant Sensor Dashboard                          │
├─────────────────────────────────────────────────────────────┤
│  Plant: ZAWAR MINES  |  Fuel: Coal  |  Capacity: 80 MW     │
├─────────────────────────────────────────────────────────────┤
│  📈 Sensor Trends                                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Temperature Chart (Line showing values over time)      │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Vibration Chart (Line showing values over time)        │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Pressure Chart (Line showing values over time)         │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  🚨 Anomalies Detected                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Normal: 950  |  Anomalies: 50  (Bar chart)             │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Anomaly Records Table (List of all problems found)     │ │
│  └─────────────────────────────────────────────────────────┘ │
│  [📥 Download Anomaly Data] Button                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 **Technical Deep Dive**

### **Isolation Forest Algorithm:**
```
How it works:
1. 🎲 Randomly picks a sensor value
2. 🎲 Randomly picks a split point  
3. 🔀 Separates data into two groups
4. 🔄 Repeats this many times
5. 📊 Normal data gets separated quickly
6. 🚨 Anomalous data takes longer to isolate
7. 🎯 Uses this to detect outliers
```

### **Why This Works:**
- **Normal sensor readings** follow predictable patterns
- **Anomalous readings** are different and "stand out"
- **Isolation Forest** is good at finding these outliers
- **5% contamination** means we expect 5% of readings to be anomalies

---

## 🎉 **Summary for Newbies**

**What we built:**
1. 📊 **Data Generator**: Creates realistic power plant sensor data
2. 🤖 **AI Trainer**: Teaches computer what normal looks like  
3. 🔍 **Anomaly Detector**: Finds problems automatically
4. 📱 **Dashboard**: Shows everything in a nice website

**Why it's useful:**
- 🚨 **Early Warning**: Catches problems before they become disasters
- ⏰ **24/7 Monitoring**: Never sleeps, always watching
- 📊 **Data Analysis**: Shows trends and patterns over time
- 💾 **Record Keeping**: Saves all findings for analysis

**Real-world impact:**
- Prevents power plant shutdowns
- Saves money on repairs
- Keeps workers safe
- Ensures reliable electricity supply

**This is MLOps in action:**
- **ML** = Machine Learning (the AI that detects anomalies)
- **Ops** = Operations (deploying and monitoring the system)
- **Pipeline** = The complete flow from data to insights

---

## 🚀 **Next Steps for Learning**

1. **Run the project** - Follow the steps above
2. **Explore the dashboard** - Click around and see what you find
3. **Modify the code** - Try changing anomaly detection sensitivity
4. **Add new sensors** - Include more sensor types
5. **Learn more** - Study Isolation Forest and other ML algorithms

**🎊 Congratulations! You now understand a complete MLOps system for thermal power plant monitoring!**
