# Quick Start Guide - MLOps Thermal Plant Monitor

This guide will help you set up and run the thermal plant monitoring system step by step.

## Prerequisites

- **Python 3.9+** (tested with Python 3.12+)
- **Git** (for cloning the repository)
- **8GB RAM minimum** (for ML models)
- **Virtual environment** (recommended)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/omendra02/mlops-thermal-plant.git
cd mlops-thermal-plant
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages (this may take 5-10 minutes)
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The installation includes large packages like TensorFlow and PyTorch. Be patient!

### Step 4: Generate Sample Data

Before training models, you need data:

```bash
python3 examples/generate_data.py
```

**Expected Output:**
```
Filtered 340 Indian thermal plants.
Saved to data/india_thermal_plants.csv
Simulating data for: CHANDRAPUR_Coal | Fuel: Coal | Capacity: 2920.0 MW
Synthetic sensor data saved to data/sensor_data.csv
```

### Step 5: Configure Settings (Optional)

For production use, copy and configure the example files:

```bash
cp config/database_config.yaml.example config/database_config.yaml
cp config/mqtt_config.yaml.example config/mqtt_config.yaml
```

Then edit these files with your actual credentials.

### Step 6: Train Models

Train the anomaly detection models:

```bash
python3 scripts/train_models.py
```

**What this does:**
- Loads sensor data from `data/sensor_data.csv`
- Trains Isolation Forest model
- Trains LSTM Autoencoder model
- Creates Ensemble model
- Saves models to `model/` directory
- Logs experiments to MLflow

**Expected time:** 2-5 minutes

### Step 7: Start Dashboard

Launch the Streamlit dashboard:

```bash
python3 scripts/start_dashboard.py
```

Or directly:

```bash
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
```

**Access the dashboard:**
- Open browser: http://localhost:8501

## Verification Tests

### Test 1: Check Package Installation

```bash
python3 -c "import pandas, numpy, sklearn, streamlit; print('✓ Core packages OK')"
```

### Test 2: Run Unit Tests

```bash
pytest tests/
```

### Test 3: Check Data Generation

```bash
ls -lh data/sensor_data.csv
```

Should show a file with sensor data.

### Test 4: Verify Models

After training:

```bash
ls -lh model/
```

Should show `.pkl` or `.h5` model files.

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Start all services
docker-compose up -d

# Access services:
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5001
# - Grafana: http://localhost:3000
```

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**
```bash
source venv/bin/activate  # Make sure venv is activated
pip install -r requirements.txt
```

### Issue 2: Data File Not Found

**Problem:** `FileNotFoundError: data/sensor_data.csv`

**Solution:**
```bash
python3 examples/generate_data.py
```

### Issue 3: Port Already in Use

**Problem:** `Address already in use` when starting dashboard

**Solution:**
```bash
# Use a different port
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py --server.port 8502
```

### Issue 4: Out of Memory

**Problem:** System runs out of memory during training

**Solution:**
- Close other applications
- Reduce batch size in `config/model_config.yaml`
- Use a smaller dataset

### Issue 5: TensorFlow/PyTorch Installation Issues

**Problem:** TensorFlow or PyTorch fails to install

**Solution:**
```bash
# Install without deep learning (use only Isolation Forest)
pip install pandas numpy scikit-learn streamlit mlflow joblib pyyaml
```

## Minimal Setup (Without Deep Learning)

If you don't need LSTM Autoencoder:

```bash
# Install minimal requirements
pip install pandas numpy scikit-learn streamlit joblib pyyaml paho-mqtt plotly

# Generate data
python3 examples/generate_data.py

# Train only Isolation Forest (modify train_models.py to skip LSTM)
```

## Project Structure Quick Reference

```
mlops-thermal-plant/
├── mlops_thermal_plant/     # Main Python package
│   ├── core/               # ML models & data processing
│   ├── iot/                # MQTT & sensor simulation
│   └── dashboard/          # Streamlit dashboard
├── config/                 # Configuration files
├── data/                   # Data directory (gitignored)
├── model/                  # Trained models (gitignored)
├── examples/               # Example scripts
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Next Steps

1. **Explore the Dashboard**: Visualize sensor data and anomalies
2. **Read Documentation**: Check `docs/` for detailed guides
3. **Customize Models**: Edit `config/model_config.yaml`
4. **Add IoT Integration**: Configure MQTT in `config/mqtt_config.yaml`
5. **Deploy to Production**: Use Docker Compose

## Key Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Generate data
python3 examples/generate_data.py

# Train models
python3 scripts/train_models.py

# Start dashboard
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py

# Run tests
pytest tests/

# View MLflow experiments
mlflow ui --port 5001
```

## Support

- **GitHub Issues**: https://github.com/omendra02/mlops-thermal-plant/issues
- **Documentation**: See `docs/` directory
- **Email**: omendra26tomar@gmail.com

## What's Working

✅ Data generation script
✅ Project structure
✅ Configuration system
✅ Package imports
✅ Basic testing framework

## What Needs Testing

⚠️ Model training (requires all dependencies installed)
⚠️ Dashboard (requires streamlit and dependencies)
⚠️ MQTT integration (requires broker)
⚠️ MLflow tracking (requires mlflow)

---

**Happy Monitoring! 🏭**
