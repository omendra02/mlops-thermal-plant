# 🧪 Testing & Deployment Guide for Thermal Plant MLOps

## 📋 **Current Project Status**

### ✅ **What's Working:**
1. **Basic Data Generation** - ✅ Working
2. **Model Training** - ✅ Working (Isolation Forest)
3. **Anomaly Prediction** - ✅ Working
4. **Basic Dashboard** - ✅ Working (Streamlit)
5. **Project Structure** - ✅ Complete

### ⚠️ **What Needs Setup:**
1. **Advanced ML Models** - Need TensorFlow/MLflow installation
2. **IoT Integration** - Need MQTT broker setup
3. **Database Integration** - Need PostgreSQL/Redis setup
4. **Docker Deployment** - Need Docker setup

---

## 🧪 **Testing Guide**

### **1. Basic Functionality Test (Current Working State)**

```bash
# Navigate to project directory
cd /Users/facets/Desktop/mlops-thermal-plant

# Activate virtual environment
source .venv/bin/activate

# Test 1: Data Generation
python data/generate_data.py
# Expected: ✅ Should generate sensor_data.csv with 1000+ rows

# Test 2: Model Training
python src/train.py
# Expected: ✅ Should create model/isolation_forest.pkl

# Test 3: Anomaly Prediction
python src/predict.py
# Expected: ✅ Should create sensor_data_with_anomalies.csv

# Test 4: Dashboard (Basic)
streamlit run dashboard.py
# Expected: ✅ Should open browser at http://localhost:8501
```

### **2. Advanced Functionality Test (After Full Setup)**

```bash
# Test 5: Advanced Dashboard
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
# Expected: ⚠️ Needs TensorFlow, MLflow, etc.

# Test 6: IoT Integration
python -m mlops_thermal_plant.iot.sensor_simulator
# Expected: ⚠️ Needs MQTT broker running

# Test 7: MLflow Integration
python scripts/train_models.py
# Expected: ⚠️ Needs MLflow server running
```

### **3. Docker Deployment Test**

```bash
# Test 8: Docker Build
docker build -t thermal-plant-mlops .
# Expected: ⚠️ Needs Docker installed

# Test 9: Docker Compose
docker-compose up -d
# Expected: ⚠️ Needs all services configured
```

---

## 🚀 **Deployment Guide**

### **Option 1: Quick Start (Current Working State)**

```bash
# 1. Setup environment
cd /Users/facets/Desktop/mlops-thermal-plant
source .venv/bin/activate

# 2. Generate data and train model
python data/generate_data.py
python src/train.py
python src/predict.py

# 3. Start dashboard
streamlit run dashboard.py
```

**Access:** http://localhost:8501

### **Option 2: Full MLOps Setup (Advanced)**

#### **Step 1: Install Advanced Dependencies**
```bash
# Install TensorFlow, MLflow, and other advanced packages
pip install tensorflow>=2.10.0
pip install mlflow>=2.5.0
pip install paho-mqtt>=1.6.0
pip install plotly>=5.10.0
pip install pyyaml>=6.0
```

#### **Step 2: Setup Databases**
```bash
# Option A: Use Docker for databases only
docker-compose up -d postgres redis influxdb mqtt

# Option B: Install locally
# PostgreSQL, Redis, InfluxDB, MQTT broker
```

#### **Step 3: Start Full Stack**
```bash
# Start all services
docker-compose up -d

# Or start individual components
python scripts/setup.sh
```

### **Option 3: Production Deployment**

#### **Cloud Deployment (AWS/Azure/GCP)**
```bash
# 1. Build production image
docker build -t thermal-plant-mlops:latest .

# 2. Push to container registry
docker tag thermal-plant-mlops:latest your-registry/thermal-plant-mlops:latest
docker push your-registry/thermal-plant-mlops:latest

# 3. Deploy to cloud
# Use Kubernetes, ECS, or similar
```

---

## 🔍 **Testing Checklist**

### **Basic Tests (Current State)**
- [ ] ✅ Data generation works
- [ ] ✅ Model training works
- [ ] ✅ Anomaly prediction works
- [ ] ✅ Basic dashboard loads
- [ ] ✅ Anomalies are displayed correctly

### **Advanced Tests (Full Setup)**
- [ ] ⚠️ IoT sensor simulation works
- [ ] ⚠️ MQTT data ingestion works
- [ ] ⚠️ LSTM model training works
- [ ] ⚠️ Ensemble model works
- [ ] ⚠️ MLflow tracking works
- [ ] ⚠️ Advanced dashboard works
- [ ] ⚠️ Real-time data updates work

### **Integration Tests**
- [ ] ⚠️ End-to-end data flow works
- [ ] ⚠️ Model retraining works
- [ ] ⚠️ Database connections work
- [ ] ⚠️ Docker containers start
- [ ] ⚠️ All services communicate

---

## 🚨 **What's Missing & Needs to be Fixed**

### **1. Missing Dependencies**
```bash
# Install these for full functionality
pip install tensorflow>=2.10.0
pip install mlflow>=2.5.0
pip install paho-mqtt>=1.6.0
pip install plotly>=5.10.0
pip install pyyaml>=6.0
pip install psycopg2-binary>=2.9.0
pip install redis>=4.3.0
pip install influxdb-client>=1.28.0
```

### **2. Missing Services**
- **MQTT Broker**: Eclipse Mosquitto
- **PostgreSQL**: Database for metadata
- **Redis**: Caching layer
- **InfluxDB**: Time-series data
- **MLflow Server**: Experiment tracking

### **3. Missing Configuration**
- **Environment Variables**: Database connections
- **MQTT Credentials**: Broker authentication
- **Model Artifacts**: Pre-trained advanced models

### **4. Missing Testing**
- **Unit Tests**: Individual component tests
- **Integration Tests**: End-to-end workflow tests
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and authorization

---

## 🛠️ **Quick Fix Commands**

### **Fix 1: Install Missing Dependencies**
```bash
cd /Users/facets/Desktop/mlops-thermal-plant
source .venv/bin/activate
pip install tensorflow mlflow paho-mqtt plotly pyyaml
```

### **Fix 2: Setup Basic MQTT Broker**
```bash
# Install Mosquitto MQTT broker
brew install mosquitto  # macOS
# or
sudo apt-get install mosquitto mosquitto-clients  # Ubuntu

# Start broker
mosquitto -c config/mosquitto.conf -v
```

### **Fix 3: Test Advanced Dashboard**
```bash
# After installing dependencies
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
```

### **Fix 4: Setup Docker Services**
```bash
# Start only essential services
docker-compose up -d postgres redis mqtt
```

---

## 📊 **Performance Testing**

### **Load Testing**
```bash
# Test with large datasets
python -c "
import pandas as pd
import numpy as np

# Generate large dataset
n_samples = 100000
data = pd.DataFrame({
    'temperature': np.random.normal(500, 10, n_samples),
    'vibration': np.random.normal(1.0, 0.2, n_samples),
    'pressure': np.random.normal(150, 5, n_samples)
})

# Test model performance
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
%time model.fit(data[['temperature', 'vibration']])
%time predictions = model.predict(data[['temperature', 'vibration']])
"
```

### **Memory Testing**
```bash
# Monitor memory usage
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

---

## 🎯 **Success Criteria**

### **Basic Success (Current State)**
- ✅ Dashboard loads and displays data
- ✅ Anomalies are detected and shown
- ✅ Model training completes successfully
- ✅ Data generation works

### **Advanced Success (Full Setup)**
- ⚠️ Real-time data ingestion works
- ⚠️ Multiple ML models train and predict
- ⚠️ MLflow experiments are tracked
- ⚠️ All services communicate properly
- ⚠️ Docker deployment works

### **Production Success**
- ⚠️ System handles production load
- ⚠️ Monitoring and alerting work
- ⚠️ Automated retraining works
- ⚠️ Security measures are in place

---

## 🚀 **Next Steps**

### **Immediate (Today)**
1. ✅ Test basic functionality (already working)
2. Install missing dependencies
3. Setup MQTT broker
4. Test advanced dashboard

### **Short Term (This Week)**
1. Setup full Docker environment
2. Implement MLflow tracking
3. Add comprehensive testing
4. Create deployment scripts

### **Long Term (This Month)**
1. Add real sensor integration
2. Implement advanced monitoring
3. Add security features
4. Optimize performance

---

## 📞 **Troubleshooting**

### **Common Issues**

**Issue 1: Module not found**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

**Issue 2: Dashboard won't start**
```bash
# Solution: Check Streamlit installation
pip install streamlit --upgrade
streamlit --version
```

**Issue 3: Model training fails**
```bash
# Solution: Check data format
python -c "
import pandas as pd
df = pd.read_csv('data/sensor_data.csv')
print(df.columns.tolist())
print(df.shape)
"
```

**Issue 4: Docker won't start**
```bash
# Solution: Check Docker installation
docker --version
docker-compose --version
```

---

**🎉 Your project is working! The basic functionality is solid. Follow this guide to add the advanced features step by step.**
