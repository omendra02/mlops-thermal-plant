# MLOps Thermal Power Plant Monitor

A comprehensive MLOps pipeline for thermal power plant monitoring and anomaly detection using IoT integration, machine learning models, and real-time dashboards.

## 🚀 Features

### Core Components
- **IoT Integration**: MQTT client for real-time sensor data ingestion
- **Advanced ML Models**: LSTM Autoencoder and Isolation Forest for anomaly detection
- **Ensemble Learning**: Combines multiple models for improved accuracy
- **Real-time Dashboard**: Streamlit-based monitoring interface
- **MLflow Integration**: Experiment tracking and model registry
- **Docker Containerization**: Full containerized deployment
- **Multi-database Support**: PostgreSQL, InfluxDB, Redis, MongoDB

### Key Capabilities
- Real-time sensor data simulation and ingestion
- Advanced anomaly detection with multiple ML algorithms
- Model training, validation, and deployment pipeline
- Experiment tracking and model versioning
- Real-time monitoring dashboard with interactive visualizations
- Automated model retraining and deployment
- Comprehensive logging and monitoring

## 📁 Project Structure

```
mlops-thermal-plant/
├── mlops_thermal_plant/          # Main Python package
│   ├── __init__.py
│   ├── core/                     # Core ML and data processing
│   │   ├── models/              # ML models (LSTM, Isolation Forest, Ensemble)
│   │   ├── data/                # Data processing and MLflow integration
│   │   └── monitoring/          # Monitoring and alerting
│   ├── iot/                     # IoT integration
│   │   ├── mqtt_client.py       # MQTT client for sensor data
│   │   └── sensor_simulator.py  # Sensor data simulator
│   └── dashboard/               # Streamlit dashboard
│       └── dashboard_app.py     # Main dashboard application
├── config/                      # Configuration files
│   ├── mqtt_config.yaml        # MQTT broker configuration
│   ├── plant_config.yaml       # Plant and sensor configurations
│   ├── model_config.yaml       # ML model configurations
│   ├── database_config.yaml    # Database connection settings
│   ├── mosquitto.conf          # MQTT broker setup
│   ├── prometheus.yml          # Prometheus monitoring config
│   └── init.sql                # Database initialization
├── data/                        # Data storage
├── models/                      # Trained model artifacts
├── experiments/                 # MLflow experiments
├── logs/                        # Application logs
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-thermal-plant
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the applications**
   - Streamlit Dashboard: http://localhost:8501
   - MLflow UI: http://localhost:5001
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Manual Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up databases** (optional - for production)
   ```bash
   # Start PostgreSQL, Redis, InfluxDB using Docker
   docker-compose up -d postgres redis influxdb mqtt
   ```

4. **Generate sample data**
   ```bash
   python data/generate_data.py
   ```

5. **Train initial models**
   ```bash
   python src/train_model.py
   ```

6. **Run the dashboard**
   ```bash
   streamlit run mlops_thermal_plant/dashboard/dashboard_app.py
   ```

## 🎯 Usage

### 1. Data Generation and Simulation

```python
from mlops_thermal_plant.iot.sensor_simulator import ThermalPlantSensorSimulator

# Create sensor simulator
simulator = ThermalPlantSensorSimulator(plant_config)

# Start simulation
simulator.start_simulation()

# Get current data
current_data = simulator.get_current_data()
```

### 2. Model Training

```python
from mlops_thermal_plant.core.models import LSTMAutoencoder, EnsembleAnomalyDetector
from mlops_thermal_plant.core.data import MLflowManager

# Initialize MLflow
mlflow_manager = MLflowManager(config)

# Train LSTM Autoencoder
with mlflow_manager.start_run():
    model = LSTMAutoencoder(model_config)
    model.fit(X_train)
    mlflow_manager.log_model(model, "lstm_autoencoder")
```

### 3. Real-time Monitoring

```python
from mlops_thermal_plant.iot.mqtt_client import ThermalPlantMQTTSubscriber

# Subscribe to sensor data
subscriber = ThermalPlantMQTTSubscriber(mqtt_config)
subscriber.start(["thermal_plant/alpha/sensors"])

# Get latest data
latest_data = subscriber.get_latest_data(100)
```

### 4. Anomaly Detection

```python
from mlops_thermal_plant.core.models import EnsembleAnomalyDetector

# Load trained ensemble model
ensemble = EnsembleAnomalyDetector(config)
ensemble.load_models("models/ensemble")

# Predict anomalies
predictions = ensemble.predict(sensor_data)
probabilities = ensemble.predict_proba(sensor_data)
```

## 🔧 Configuration

### MQTT Configuration (`config/mqtt_config.yaml`)
```yaml
mqtt:
  broker:
    host: "localhost"
    port: 1883
  topics:
    sensor_data: "thermal_plant/{plant_name}/sensors"
```

### Model Configuration (`config/model_config.yaml`)
```yaml
models:
  lstm_autoencoder:
    sequence_length: 60
    encoding_dim: 32
    epochs: 100
  isolation_forest:
    contamination: 0.05
    n_estimators: 100
```

### Plant Configuration (`config/plant_config.yaml`)
```yaml
plants:
  - plant_name: "Thermal Plant Alpha"
    fuel_type: "Coal"
    capacity_mw: 500
    sensors:
      steam_temperature:
        normal_range: [480, 520]
        unit: "°C"
```

## 📊 Dashboard Features

### Real-time Monitoring Tab
- Live sensor readings and trends
- Equipment health indicators
- Performance metrics
- Alert notifications

### Anomaly Detection Tab
- Model predictions and confidence scores
- Anomaly timeline visualization
- Model performance metrics
- Feature importance analysis

### MLflow Experiments Tab
- Experiment tracking and comparison
- Model versioning and registry
- Performance metrics visualization
- Model deployment status

### Model Training Tab
- Interactive model training interface
- Hyperparameter tuning
- Training progress monitoring
- Model evaluation results

## 🔄 MLOps Pipeline

### 1. Data Ingestion
- IoT sensors → MQTT → Data processing pipeline
- Real-time data validation and quality checks
- Data storage in multiple databases

### 2. Model Development
- Feature engineering and preprocessing
- Model training with MLflow tracking
- Cross-validation and hyperparameter tuning
- Model evaluation and selection

### 3. Model Deployment
- Model versioning and registry
- Automated deployment pipeline
- A/B testing and gradual rollout
- Performance monitoring

### 4. Monitoring & Retraining
- Real-time model performance monitoring
- Data drift detection
- Automated retraining triggers
- Model rollback capabilities

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| thermal-plant-app | 8501 | Main Streamlit application |
| mlflow | 5001 | MLflow tracking server |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| influxdb | 8086 | InfluxDB time-series database |
| mqtt | 1883 | MQTT broker |
| grafana | 3000 | Grafana monitoring dashboard |
| prometheus | 9090 | Prometheus metrics collection |
| minio | 9000 | S3-compatible storage |

## 📈 Monitoring & Alerting

### Metrics Collection
- Application performance metrics
- Model prediction accuracy
- System resource utilization
- Database performance metrics

### Alerting Rules
- Anomaly detection alerts
- Model performance degradation
- System health issues
- Data quality problems

### Visualization
- Grafana dashboards for system monitoring
- Prometheus metrics for alerting
- Custom Streamlit visualizations
- MLflow experiment tracking

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=mlops_thermal_plant tests/

# Run integration tests
pytest tests/integration/
```

## 🚀 Deployment

### Development Environment
```bash
docker-compose -f docker-compose.yml up -d
```

### Production Environment
```bash
# Use production Dockerfile
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### Environment Variables
```bash
# Database connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=thermal_plant_db
POSTGRES_USER=thermal_user
POSTGRES_PASSWORD=thermal_password

# MLflow
MLFLOW_TRACKING_URI=postgresql://user:pass@host:port/db

# MQTT
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thermal power plant domain knowledge and data simulation
- MLOps best practices and tools integration
- IoT protocols and sensor data handling
- Real-time monitoring and alerting systems

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` folder
- Review the configuration examples in `config/`

---

**Built with ❤️ for thermal power plant operations and MLOps excellence**