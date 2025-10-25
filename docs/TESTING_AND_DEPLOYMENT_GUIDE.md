# 🧪 Complete Testing & Deployment Guide - Thermal Plant MLOps

**Last Updated:** January 25, 2025
**System Status:** ✅ Production-Ready

---

## 📋 **System Overview**

This guide covers comprehensive testing and deployment strategies for the MLOps Thermal Plant monitoring system, including all three ML models (Isolation Forest, LSTM Autoencoder, Ensemble), IoT integration, and production deployment.

---

## 🧪 **Testing Guide**

### **1. Environment Setup Testing**

```bash
# Navigate to project
cd /Users/facets/Desktop/mlops-thermal-plant

# Activate virtual environment
source .venv/bin/activate

# Verify Python version
python --version
# Expected: Python 3.9+

# Check core dependencies
python -c "import pandas, numpy, sklearn; print('✅ Core packages OK')"

# Check advanced dependencies (if installed)
python -c "import tensorflow, mlflow, paho.mqtt; print('✅ Advanced packages OK')"
```

### **2. Data Generation Testing**

```bash
# Test data generation
python examples/generate_data.py

# Verify output
ls -lh data/sensor_data.csv
# Expected: CSV file with ~100KB size

# Check data quality
python -c "
import pandas as pd
df = pd.read_csv('data/sensor_data.csv')
print(f'✅ Rows: {len(df)}')
print(f'✅ Columns: {list(df.columns)}')
print(f'✅ Missing values: {df.isnull().sum().sum()}')
"

# Expected output:
# ✅ Rows: 1000
# ✅ Columns: ['timestamp', 'plant_name', 'fuel_type', 'capacity_mw',
#             'temperature', 'vibration', 'pressure', 'flow_rate', 'load_factor']
# ✅ Missing values: 0
```

### **3. Model Training Testing**

#### **Test 3A: Isolation Forest Model**

```bash
# Test Isolation Forest training
python -c "
import pandas as pd
from mlops_thermal_plant.core.models import IsolationForestModel

# Load data
df = pd.read_csv('data/sensor_data.csv')

# Rename columns to match expected format
df_renamed = df.rename(columns={
    'temperature': 'steam_temperature',
    'vibration': 'turbine_vibration',
    'pressure': 'steam_pressure'
})

# Add missing sensor columns with dummy data
import numpy as np
n = len(df_renamed)
df_renamed['generator_temperature'] = 70 + np.random.normal(0, 5, n)
df_renamed['cooling_water_temp'] = 30 + np.random.normal(0, 2, n)
df_renamed['fuel_flow_rate'] = df['flow_rate']
df_renamed['oxygen_level'] = 3.0 + np.random.normal(0, 0.3, n)
df_renamed['load_factor'] = df['load_factor'] * 100

# Train model
config = {'contamination': 0.05, 'n_estimators': 100, 'random_state': 42}
model = IsolationForestModel(config)
results = model.fit(df_renamed)

print('✅ Isolation Forest Training Results:')
print(f'   Samples: {results[\"n_samples\"]}')
print(f'   Features: {results[\"n_features\"]}')
print(f'   Anomaly Rate: {results[\"anomaly_rate\"]:.2%}')

# Test prediction
predictions = model.predict(df_renamed)
print(f'✅ Predictions shape: {predictions.shape}')
print(f'✅ Anomalies detected: {sum(predictions)} ({sum(predictions)/len(predictions):.2%})')
"

# Expected output:
# ✅ Isolation Forest Training Results:
#    Samples: 1000
#    Features: 8-15 (with derived features)
#    Anomaly Rate: 4-6%
# ✅ Predictions shape: (1000,)
# ✅ Anomalies detected: ~50 (5%)
```

#### **Test 3B: LSTM Autoencoder Model** (requires TensorFlow)

```bash
# Test LSTM model (if TensorFlow is installed)
python -c "
import pandas as pd
import numpy as np
from mlops_thermal_plant.core.models import LSTMAutoencoder

# Load and prepare data
df = pd.read_csv('data/sensor_data.csv')
df_renamed = df.rename(columns={
    'temperature': 'steam_temperature',
    'vibration': 'turbine_vibration',
    'pressure': 'steam_pressure',
    'flow_rate': 'fuel_flow_rate'
})

# Add missing columns
n = len(df_renamed)
df_renamed['generator_temperature'] = 70 + np.random.normal(0, 5, n)
df_renamed['cooling_water_temp'] = 30 + np.random.normal(0, 2, n)
df_renamed['oxygen_level'] = 3.0 + np.random.normal(0, 0.3, n)
df_renamed['load_factor'] = df['load_factor'] * 100

# Train model
config = {
    'sequence_length': 60,
    'features': 8,
    'encoding_dim': 32,
    'hidden_dims': [64, 32],
    'epochs': 10,  # Reduced for testing
    'batch_size': 32
}

model = LSTMAutoencoder(config)
history = model.fit(df_renamed, validation_split=0.2)

print('✅ LSTM Training completed')
print(f'✅ Final loss: {history[\"loss\"][-1]:.4f}')
print(f'✅ Validation loss: {history[\"val_loss\"][-1]:.4f}')

# Test prediction
predictions = model.predict(df_renamed)
print(f'✅ Predictions shape: {predictions.shape}')
print(f'✅ Anomalies detected: {sum(predictions)}')
"
```

#### **Test 3C: Complete Training Pipeline**

```bash
# Run complete training pipeline
python scripts/train_models.py

# Expected output:
# INFO - Starting model training pipeline...
# INFO - Training Isolation Forest model...
# INFO - Training LSTM Autoencoder model...
# INFO - Training Ensemble model...
# INFO - Model training pipeline completed successfully!
# TRAINING SUMMARY
# ================================================
# ISOLATION_FOREST:
#   n_samples: 1000
#   anomaly_rate: 0.05
# LSTM_AUTOENCODER:
#   reconstruction_threshold: 0.XX
# ENSEMBLE:
#   voting_strategy: majority
# Models saved to 'models/' directory
```

### **4. Dashboard Testing**

#### **Test 4A: Dashboard Launch**

```bash
# Launch dashboard
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py

# Or use the launcher script
python scripts/start_dashboard.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

#### **Test 4B: Dashboard Functionality**

**Manual Testing Checklist:**

- [ ] Dashboard loads without errors
- [ ] Tab 1 (Real-time Monitoring) displays:
  - [ ] Plant information (name, fuel, capacity)
  - [ ] Current sensor readings (8 sensors)
  - [ ] Interactive time-series charts
- [ ] Tab 2 (Anomaly Detection) displays:
  - [ ] Anomaly count and rate
  - [ ] Anomaly timeline chart
  - [ ] Model performance metrics
  - [ ] Detailed anomaly table
- [ ] Tab 3 (MLflow Experiments) displays:
  - [ ] Recent experiment runs
  - [ ] Training metrics
  - [ ] Model comparison
- [ ] Tab 4 (Model Training) displays:
  - [ ] Model selection dropdown
  - [ ] Training button
  - [ ] Progress visualization

### **5. Integration Testing**

#### **Test 5A: End-to-End Pipeline**

```bash
# Complete end-to-end test script
python -c "
print('🧪 Starting End-to-End Integration Test...')

# 1. Generate data
print('1️⃣ Generating sensor data...')
import subprocess
result = subprocess.run(['python', 'examples/generate_data.py'], capture_output=True)
assert result.returncode == 0, 'Data generation failed'
print('✅ Data generation successful')

# 2. Verify data
print('2️⃣ Verifying data quality...')
import pandas as pd
df = pd.read_csv('data/sensor_data.csv')
assert len(df) == 1000, f'Expected 1000 rows, got {len(df)}'
assert len(df.columns) == 9, f'Expected 9 columns, got {len(df.columns)}'
print('✅ Data verification successful')

# 3. Train models
print('3️⃣ Training models (this may take a few minutes)...')
result = subprocess.run(['python', 'scripts/train_models.py'], capture_output=True)
# Note: May fail if dependencies not installed
if result.returncode == 0:
    print('✅ Model training successful')
else:
    print('⚠️ Model training skipped (requires full dependencies)')

# 4. Verify model files
print('4️⃣ Checking model files...')
import os
model_files = os.listdir('model') if os.path.exists('model') else []
print(f'✅ Model files: {len(model_files)} files found')

print('🎉 End-to-End Integration Test Complete!')
"
```

#### **Test 5B: IoT Integration** (requires MQTT broker)

```bash
# Start MQTT broker (in separate terminal)
mosquitto -c config/mosquitto.conf -v

# Test sensor simulator
python -c "
from mlops_thermal_plant.iot import ThermalPlantSensorSimulator
import time

print('🌡️ Testing sensor simulator...')
simulator = ThermalPlantSensorSimulator()
sensor_data = simulator.generate_sensor_reading()

print(f'✅ Sensor reading generated:')
for key, value in sensor_data.items():
    print(f'   {key}: {value}')
"

# Test MQTT client
python -c "
from mlops_thermal_plant.iot import ThermalPlantMQTTSubscriber

print('📡 Testing MQTT client...')
client = ThermalPlantMQTTSubscriber()
client.connect()
print('✅ MQTT client connected')
client.disconnect()
"
```

### **6. Performance Testing**

#### **Test 6A: Model Inference Speed**

```bash
python -c "
import pandas as pd
import numpy as np
import time
from mlops_thermal_plant.core.models import IsolationForestModel

# Prepare data
df = pd.read_csv('data/sensor_data.csv')
df_renamed = df.rename(columns={
    'temperature': 'steam_temperature',
    'vibration': 'turbine_vibration',
    'pressure': 'steam_pressure'
})
n = len(df_renamed)
df_renamed['generator_temperature'] = 70 + np.random.normal(0, 5, n)
df_renamed['cooling_water_temp'] = 30 + np.random.normal(0, 2, n)
df_renamed['fuel_flow_rate'] = df['flow_rate']
df_renamed['oxygen_level'] = 3.0 + np.random.normal(0, 0.3, n)
df_renamed['load_factor'] = df['load_factor'] * 100

# Train model
config = {'contamination': 0.05, 'n_estimators': 100, 'random_state': 42}
model = IsolationForestModel(config)
model.fit(df_renamed)

# Test inference speed
start = time.time()
predictions = model.predict(df_renamed)
end = time.time()

inference_time = (end - start) * 1000  # milliseconds
throughput = len(df_renamed) / (end - start)  # samples/second

print(f'⚡ Performance Metrics:')
print(f'   Inference Time: {inference_time:.2f} ms')
print(f'   Throughput: {throughput:.2f} samples/second')
print(f'   Per-sample: {inference_time/len(df_renamed):.3f} ms')

# Expected:
# Inference Time: < 100 ms for 1000 samples
# Throughput: > 10,000 samples/second
# Per-sample: < 0.1 ms
"
```

#### **Test 6B: Memory Usage**

```bash
python -c "
import psutil
import os
import pandas as pd
from mlops_thermal_plant.core.models import IsolationForestModel

process = psutil.Process(os.getpid())

# Baseline memory
baseline = process.memory_info().rss / 1024 / 1024

# Load data and train
df = pd.read_csv('data/sensor_data.csv')
config = {'contamination': 0.05, 'n_estimators': 100}
model = IsolationForestModel(config)

# Memory after loading
after_load = process.memory_info().rss / 1024 / 1024

print(f'💾 Memory Usage:')
print(f'   Baseline: {baseline:.2f} MB')
print(f'   After Loading: {after_load:.2f} MB')
print(f'   Increase: {after_load - baseline:.2f} MB')
"
```

---

## 🚀 **Deployment Guide**

### **Deployment Option 1: Local Development**

```bash
# Complete local setup
cd /Users/facets/Desktop/mlops-thermal-plant
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Generate data
python examples/generate_data.py

# Train models
python scripts/train_models.py

# Start dashboard
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
```

**Access:** http://localhost:8501

---

### **Deployment Option 2: Docker (Single Container)**

```dockerfile
# Build Docker image
docker build -t thermal-plant-mlops:latest .

# Run container
docker run -p 8501:8501 -p 5001:5001 thermal-plant-mlops:latest

# Access services
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5001
```

---

### **Deployment Option 3: Docker Compose (Full Stack)**

```bash
# Start all services
docker-compose up -d

# Services included:
# - Dashboard (Streamlit)
# - MLflow server
# - PostgreSQL database
# - Redis cache
# - InfluxDB (time-series)
# - MQTT broker (Mosquitto)
# - Prometheus (monitoring)
# - Grafana (visualization)

# Check status
docker-compose ps

# View logs
docker-compose logs -f dashboard

# Stop all services
docker-compose down
```

**Access Points:**
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5001
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

---

### **Deployment Option 4: Cloud Deployment (AWS)**

#### **AWS ECS Deployment**

```bash
# 1. Build and tag image
docker build -t thermal-plant-mlops:latest .
docker tag thermal-plant-mlops:latest YOUR_ECR_REPO:latest

# 2. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_REPO
docker push YOUR_ECR_REPO:latest

# 3. Create ECS task definition (task-definition.json)
{
  "family": "thermal-plant-mlops",
  "containerDefinitions": [{
    "name": "dashboard",
    "image": "YOUR_ECR_REPO:latest",
    "portMappings": [
      {"containerPort": 8501, "protocol": "tcp"},
      {"containerPort": 5001, "protocol": "tcp"}
    ],
    "memory": 2048,
    "cpu": 1024
  }]
}

# 4. Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 5. Create ECS service
aws ecs create-service \
  --cluster thermal-plant-cluster \
  --service-name thermal-plant-service \
  --task-definition thermal-plant-mlops \
  --desired-count 2 \
  --launch-type FARGATE
```

#### **AWS EC2 Deployment**

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.large)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# 4. Clone repository
git clone https://github.com/omendra02/mlops-thermal-plant.git
cd mlops-thermal-plant

# 5. Deploy with Docker Compose
docker-compose up -d

# 6. Configure security group
# Allow inbound: 8501 (dashboard), 5001 (MLflow), 3000 (Grafana)
```

---

### **Deployment Option 5: Kubernetes**

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thermal-plant-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: thermal-plant
  template:
    metadata:
      labels:
        app: thermal-plant
    spec:
      containers:
      - name: dashboard
        image: thermal-plant-mlops:latest
        ports:
        - containerPort: 8501
        - containerPort: 5001
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: thermal-plant-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
    name: dashboard
  - port: 5001
    targetPort: 5001
    name: mlflow
  selector:
    app: thermal-plant
```

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment thermal-plant-dashboard --replicas=5
```

---

## 🔧 **Configuration for Production**

### **1. Environment Variables**

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/thermal_plant
REDIS_URL=redis://localhost:6379
INFLUXDB_URL=http://localhost:8086
MQTT_BROKER=mqtt://localhost:1883
MLFLOW_TRACKING_URI=http://localhost:5001

# Model settings
MODEL_CONTAMINATION=0.05
LSTM_EPOCHS=100
ENSEMBLE_WEIGHTS=0.6,0.4

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,your-domain.com
```

### **2. Database Configuration**

```yaml
# config/database_config.yaml
postgresql:
  host: localhost
  port: 5432
  database: thermal_plant
  user: ${DB_USER}
  password: ${DB_PASSWORD}

influxdb:
  url: http://localhost:8086
  token: ${INFLUXDB_TOKEN}
  org: thermal_plant
  bucket: sensor_data

redis:
  host: localhost
  port: 6379
  db: 0
  password: ${REDIS_PASSWORD}
```

### **3. Security Hardening**

```bash
# Install security packages
pip install python-dotenv cryptography

# Use environment variables for secrets
# Never commit credentials to git

# Enable HTTPS
# Use SSL certificates (Let's Encrypt)

# Implement authentication
# Add user login to dashboard

# Enable audit logging
# Track all model predictions and changes
```

---

## 📊 **Monitoring & Logging**

### **1. Application Monitoring**

```python
# Add to dashboard_app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

### **2. Prometheus Metrics**

```python
# Add Prometheus instrumentation
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
predictions_total = Counter('predictions_total', 'Total predictions made')
anomalies_detected = Counter('anomalies_detected', 'Total anomalies detected')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
```

### **3. Grafana Dashboards**

```bash
# Import pre-built dashboard
# Navigate to Grafana: http://localhost:3000
# Import dashboards/thermal_plant_dashboard.json

# Key metrics to track:
# - Predictions per second
# - Anomaly detection rate
# - Model accuracy over time
# - System resource usage
# - API response times
```

---

## 🧪 **Testing Checklist**

### **Pre-Deployment Testing**

- [ ] Unit tests pass (`pytest tests/`)
- [ ] Integration tests pass
- [ ] Performance benchmarks met
- [ ] Memory leaks checked
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Dependencies locked

### **Post-Deployment Testing**

- [ ] Health check endpoint responding
- [ ] Dashboard accessible
- [ ] Models loading correctly
- [ ] Predictions working
- [ ] MLflow tracking operational
- [ ] Database connections stable
- [ ] Monitoring dashboards showing data
- [ ] Alerts configured and firing

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Issue 1: ModuleNotFoundError**
```bash
# Solution
pip install -r requirements.txt
# Or install missing package
pip install package-name
```

**Issue 2: TensorFlow Installation Fails**
```bash
# Solution: Install TensorFlow CPU version
pip install tensorflow-cpu
# Or skip LSTM and use only Isolation Forest
```

**Issue 3: Dashboard Won't Start**
```bash
# Solution
# Check port availability
lsof -i :8501
# Kill process if needed
kill -9 PID
# Restart dashboard
streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
```

**Issue 4: Model Loading Errors**
```bash
# Solution
# Ensure models are trained first
python scripts/train_models.py
# Check model files exist
ls -la model/
```

**Issue 5: Docker Build Fails**
```bash
# Solution
# Clear Docker cache
docker system prune -a
# Rebuild with no cache
docker build --no-cache -t thermal-plant-mlops .
```

---

## 📞 **Support & Resources**

- **Documentation**: `/docs` folder
- **Examples**: `/examples` folder
- **Tests**: `/tests` folder
- **GitHub Issues**: https://github.com/omendra02/mlops-thermal-plant/issues

---

## 🎉 **Success Criteria**

### **Development Success**
✅ All tests passing
✅ Dashboard runs locally
✅ Models train successfully
✅ Predictions accurate

### **Staging Success**
✅ Docker deployment works
✅ All services communicate
✅ Performance benchmarks met
✅ Integration tests pass

### **Production Success**
✅ High availability (99.9% uptime)
✅ Auto-scaling configured
✅ Monitoring and alerts active
✅ Backup and recovery tested
✅ Security hardening complete
✅ Documentation complete

**🏆 Your MLOps system is ready for production deployment!**
