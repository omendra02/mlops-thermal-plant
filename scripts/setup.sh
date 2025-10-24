#!/bin/bash

# Thermal Plant MLOps Setup Script

set -e

echo "🚀 Setting up Thermal Plant MLOps Environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models logs experiments config/grafana/dashboards config/grafana/datasources

# Set up Python virtual environment (if not using Docker)
if [ "$1" = "--local" ]; then
    echo "🐍 Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Generate sample data
echo "📊 Generating sample data..."
python data/generate_data.py

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

echo "✅ Setup complete!"
echo ""
echo "🌐 Access URLs:"
echo "  - Streamlit Dashboard: http://localhost:8501"
echo "  - MLflow UI: http://localhost:5001"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - InfluxDB: http://localhost:8086"
echo ""
echo "📚 Next steps:"
echo "  1. Train initial models: python src/train_model.py"
echo "  2. Run anomaly detection: python src/predict.py"
echo "  3. Access the dashboard at http://localhost:8501"
