# MLOps Thermal Power Plant Monitor

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive MLOps pipeline for real-time thermal power plant monitoring and anomaly detection using IoT integration, machine learning models, and interactive dashboards.

## Features

- **Real-time IoT Integration**: MQTT-based sensor data ingestion
- **Advanced ML Models**: Ensemble of Isolation Forest and LSTM Autoencoder for anomaly detection
- **MLflow Integration**: Complete experiment tracking, model versioning, and registry
- **Interactive Dashboard**: Streamlit-based real-time monitoring interface
- **Production Ready**: Dockerized deployment with multi-database support
- **Scalable Architecture**: Supports PostgreSQL, InfluxDB, Redis, and MongoDB

## Project Structure

```
mlops-thermal-plant/
├── mlops_thermal_plant/      # Main package
│   ├── core/                 # ML models and data processing
│   │   ├── models/          # Isolation Forest, LSTM, Ensemble
│   │   └── data/            # Data processors and MLflow manager
│   ├── iot/                 # MQTT client and sensor simulator
│   └── dashboard/           # Streamlit dashboard application
├── config/                   # Configuration templates
│   ├── *.yaml.example       # Example configurations
│   ├── model_config.yaml    # ML model parameters
│   ├── plant_config.yaml    # Plant sensor configurations
│   └── prometheus.yml       # Monitoring setup
├── scripts/                  # Utility scripts
│   ├── train_models.py      # Model training pipeline
│   └── start_dashboard.py   # Dashboard launcher
├── tests/                    # Unit and integration tests
├── docs/                     # Additional documentation
├── examples/                 # Example usage and data generation
├── docker-compose.yml        # Docker services orchestration
├── Dockerfile                # Container definition
└── setup.py                  # Package installation

```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for full stack)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/omendra02/mlops-thermal-plant.git
cd mlops-thermal-plant
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure settings**
```bash
# Copy example configs and update with your credentials
cp config/database_config.yaml.example config/database_config.yaml
cp config/mqtt_config.yaml.example config/mqtt_config.yaml
```

4. **Generate sample data** (for testing)
```bash
python examples/generate_data.py
```

5. **Train models**
```bash
python scripts/train_models.py
```

6. **Start dashboard**
```bash
python scripts/start_dashboard.py
```

### Docker Deployment

Run the complete stack with all services:

```bash
docker-compose up -d
```

Access the applications:
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5001
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Architecture

### Data Flow

1. **Data Ingestion**: IoT sensors → MQTT Broker → Data Processor
2. **Storage**: Multi-tier storage (PostgreSQL, InfluxDB, Redis)
3. **ML Pipeline**: Feature Engineering → Model Training → Prediction
4. **Monitoring**: Real-time dashboard + Prometheus + Grafana
5. **MLOps**: MLflow experiment tracking and model registry

### ML Models

- **Isolation Forest**: Fast, unsupervised anomaly detection
- **LSTM Autoencoder**: Deep learning-based pattern recognition
- **Ensemble Model**: Combines both models for improved accuracy

## Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
isolation_forest:
  contamination: 0.05
  n_estimators: 100

lstm_autoencoder:
  sequence_length: 50
  encoding_dim: 32
  epochs: 100
  batch_size: 32
```

### Plant Configuration (`config/plant_config.yaml`)

Configure sensor types, thresholds, and monitoring parameters.

## Development

### Install in development mode

```bash
pip install -e .
```

### Run tests

```bash
pytest tests/
```

### Code quality

```bash
black mlops_thermal_plant/
flake8 mlops_thermal_plant/
mypy mlops_thermal_plant/
```

## Usage Examples

### Train Models

```python
from mlops_thermal_plant.core.models import EnsembleModel
from mlops_thermal_plant.core.data import DataProcessor

# Load and process data
processor = DataProcessor()
X_train, X_test = processor.load_and_prepare_data()

# Train ensemble model
model = EnsembleModel()
model.train(X_train)
predictions = model.predict(X_test)
```

### Start IoT Simulation

```python
from mlops_thermal_plant.iot import SensorSimulator, MQTTClient

# Initialize components
simulator = SensorSimulator()
mqtt_client = MQTTClient()

# Start simulation
simulator.start()
mqtt_client.subscribe("thermal_plant/sensors")
```

## Monitoring & Alerts

The system provides comprehensive monitoring:
- Real-time sensor metrics visualization
- Anomaly detection alerts
- Model performance tracking
- System health dashboards

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Beginner's Guide](docs/BEGINNER_GUIDE.md)
- [Complete System Explanation](docs/COMPLETE_EXPLANATION.md)
- [Testing & Deployment Guide](docs/TESTING_AND_DEPLOYMENT_GUIDE.md)
- [Code Structure Diagram](docs/CODE_STRUCTURE_DIAGRAM.txt)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for industrial IoT monitoring and predictive maintenance
- Uses industry-standard MLOps practices
- Inspired by real-world thermal power plant operations

## Contact

Omendra Tomar - [@omendra02](https://github.com/omendra02)

Project Link: [https://github.com/omendra02/mlops-thermal-plant](https://github.com/omendra02/mlops-thermal-plant)
