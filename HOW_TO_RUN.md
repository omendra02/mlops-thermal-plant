# How to Run - MLOps Thermal Plant Monitor

**Quick Start Guide for Complete Beginners**

---

## Prerequisites

- Computer with macOS, Linux, or Windows
- Python 3.9 or higher installed
- Internet connection (for initial setup)
- 8GB RAM recommended

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Get the Code

```bash
# Download from GitHub
git clone https://github.com/omendra02/mlops-thermal-plant.git
cd mlops-thermal-plant
```

### Step 2: Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # macOS/Linux
# OR
venv\Scripts\activate             # Windows
```

### Step 3: Install Packages

```bash
# Install dependencies (takes 5-10 minutes)
pip install pandas numpy scikit-learn streamlit joblib plotly
```

### Step 4: Generate Data

```bash
# Create sample sensor data
python3 examples/generate_data.py
```

**Output:**
```
Filtered 340 Indian thermal plants.
Simulating data for: CHANDRAPUR_Coal | Fuel: Coal | Capacity: 2920.0 MW
Synthetic sensor data saved to data/sensor_data.csv
```

### Step 5: Run Tests

```bash
# Verify everything works
python3 comprehensive_test.py
```

**Expected:** All tests pass ✅

### Step 6: Start Dashboard

```bash
# Launch interactive dashboard
streamlit run test_dashboard_simple.py
```

**Open browser:** http://localhost:8501

---

## 📊 What You'll See

### Dashboard Features:
1. **Plant Metrics** - Capacity, fuel type, status
2. **Current Readings** - Temperature, vibration, pressure, flow rate, load factor
3. **Anomaly Detection** - Real-time anomaly identification
4. **Time Series Plots** - Interactive sensor data visualization
5. **Statistics** - Data summary and analysis
6. **Download** - Export anomaly reports

---

## 🎯 Usage Examples

### Example 1: Generate Data
```bash
python3 examples/generate_data.py
```

### Example 2: Train Model
```bash
python3 comprehensive_test.py
```

### Example 3: View Dashboard
```bash
streamlit run test_dashboard_simple.py
```

### Example 4: Check Anomalies
```bash
cat output/detected_anomalies.csv
```

---

## 📁 Important Files

### Input Files:
- `data/sensor_data.csv` - Generated sensor readings
- `config/*.yaml` - Configuration files

### Output Files:
- `model/isolation_forest_trained.pkl` - Trained model
- `output/detected_anomalies.csv` - Anomaly report
- `output/training_summary.json` - Training statistics

### Test Scripts:
- `simple_test.py` - Quick verification
- `comprehensive_test.py` - Full system test
- `test_dashboard_simple.py` - Dashboard app

---

## 🔧 Troubleshooting

### Problem 1: "command not found: python3"
**Solution:**
```bash
python --version  # Try without the 3
# OR install Python from python.org
```

### Problem 2: "No module named 'pandas'"
**Solution:**
```bash
source venv/bin/activate  # Activate environment first
pip install -r requirements.txt
```

### Problem 3: "Data file not found"
**Solution:**
```bash
python3 examples/generate_data.py  # Generate data first
```

### Problem 4: "Port already in use"
**Solution:**
```bash
# Use different port
streamlit run test_dashboard_simple.py --server.port 8502
```

### Problem 5: Dashboard won't start
**Solution:**
```bash
pip install streamlit plotly
streamlit run test_dashboard_simple.py
```

---

## 📖 Step-by-Step Workflows

### Workflow 1: First Time Setup
```bash
# 1. Clone repository
git clone https://github.com/omendra02/mlops-thermal-plant.git
cd mlops-thermal-plant

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate

# 3. Install packages
pip install pandas numpy scikit-learn streamlit joblib plotly

# 4. Generate data
python3 examples/generate_data.py

# 5. Test system
python3 comprehensive_test.py

# 6. Start dashboard
streamlit run test_dashboard_simple.py
```

### Workflow 2: Daily Use
```bash
# 1. Navigate to project
cd mlops-thermal-plant

# 2. Activate environment
source venv/bin/activate

# 3. Start dashboard
streamlit run test_dashboard_simple.py
```

### Workflow 3: Generate New Data
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Generate fresh data
python3 examples/generate_data.py

# 3. Retrain model
python3 comprehensive_test.py

# 4. View in dashboard
streamlit run test_dashboard_simple.py
```

---

## 🎓 Understanding the System

### Data Flow:
```
1. Generate Data
   examples/generate_data.py
   → Creates data/sensor_data.csv

2. Train Model
   comprehensive_test.py
   → Trains anomaly detector
   → Saves model/isolation_forest_trained.pkl

3. Detect Anomalies
   Model identifies unusual patterns
   → Saves output/detected_anomalies.csv

4. Visualize
   Dashboard shows real-time monitoring
   → http://localhost:8501
```

### Key Components:
- **Data Generator** - Creates synthetic sensor data
- **Isolation Forest** - Detects anomalies in data
- **Dashboard** - Visualizes data and anomalies
- **Test Scripts** - Verify system functionality

---

## 🌟 Advanced Usage

### Full Installation (All Features):
```bash
# Install everything
pip install -r requirements.txt

# May include TensorFlow, MLflow, etc.
# Note: TensorFlow may not work on all systems
```

### Using Docker:
```bash
# Start all services
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5001
# - Grafana: http://localhost:3000
```

### Custom Configuration:
```bash
# Copy example configs
cp config/database_config.yaml.example config/database_config.yaml
cp config/mqtt_config.yaml.example config/mqtt_config.yaml

# Edit with your credentials
nano config/database_config.yaml
```

---

## 📊 Expected Results

### After Running Tests:
```
✓ Data: 1000 sensor readings
✓ Model: Trained Isolation Forest
✓ Anomalies: ~50 detected (5%)
✓ Files: Model saved, reports generated
```

### Dashboard View:
- Current sensor readings with trends
- Interactive time series plots
- Anomaly highlights in red
- Statistical summaries
- Downloadable reports

---

## 🎯 Success Criteria

You know it's working when you see:

1. ✅ Data generated: `data/sensor_data.csv` exists
2. ✅ Model trained: `model/isolation_forest_trained.pkl` exists
3. ✅ Tests pass: "TEST COMPLETE - ALL CHECKS PASSED ✓"
4. ✅ Dashboard loads: Browser shows plant metrics
5. ✅ Anomalies detected: Red markers on plots

---

## 💡 Tips & Tricks

1. **Start Simple**: Use minimal installation first
2. **Check Logs**: Read error messages carefully
3. **Virtual Environment**: Always activate before running
4. **Fresh Start**: Delete data/ and model/ to start over
5. **Documentation**: Read QUICKSTART.md for details

---

## 🆘 Getting Help

1. **Read Documentation**: Check `docs/` folder
2. **View Test Results**: See `TEST_RESULTS.md`
3. **Check Examples**: Look at test scripts
4. **GitHub Issues**: Report problems on GitHub

---

## 📝 Quick Reference

### Essential Commands:
```bash
# Activate environment
source venv/bin/activate

# Generate data
python3 examples/generate_data.py

# Run tests
python3 comprehensive_test.py

# Start dashboard
streamlit run test_dashboard_simple.py

# Stop dashboard
Ctrl+C

# Deactivate environment
deactivate
```

### Essential Files:
```
├── examples/generate_data.py      # Create data
├── comprehensive_test.py           # Test system
├── test_dashboard_simple.py        # Run dashboard
├── data/sensor_data.csv           # Input data
├── model/*.pkl                     # Trained models
└── output/detected_anomalies.csv  # Results
```

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.9+ installed (`python3 --version`)
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] Packages installed (`pip list | grep pandas`)
- [ ] Data generated (`ls data/sensor_data.csv`)
- [ ] Tests passed (`python3 comprehensive_test.py`)
- [ ] Dashboard accessible (http://localhost:8501)

---

## 🚀 Next Steps

After successful setup:

1. **Explore Dashboard** - Click around, view different sensors
2. **Read Documentation** - Check `docs/` for deep dive
3. **Customize** - Edit `config/` files for your needs
4. **Integrate** - Add real sensor data
5. **Deploy** - Use Docker for production

---

**Need Help?** Open an issue on GitHub: https://github.com/omendra02/mlops-thermal-plant/issues

**Enjoy Monitoring! ⚡**
