#!/bin/bash

# Quick Start Script for Thermal Plant MLOps
# This script sets up and runs the basic working functionality

set -e

echo "🚀 Thermal Plant MLOps - Quick Start"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   Create virtual environment: python3 -m venv .venv"
    echo "   Activate: source .venv/bin/activate"
    echo "   Install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import pandas, numpy, sklearn, joblib, streamlit" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install pandas numpy scikit-learn joblib streamlit
}

# Generate data if not exists
echo "📊 Generating sensor data..."
if [ ! -f "data/sensor_data.csv" ]; then
    python data/generate_data.py
    echo "✅ Sensor data generated"
else
    echo "✅ Sensor data already exists"
fi

# Train model if not exists
echo "🤖 Training anomaly detection model..."
if [ ! -f "model/isolation_forest.pkl" ]; then
    python src/train.py
    echo "✅ Model trained"
else
    echo "✅ Model already trained"
fi

# Run anomaly detection if not exists
echo "🔍 Running anomaly detection..."
if [ ! -f "data/sensor_data_with_anomalies.csv" ]; then
    python src/predict.py
    echo "✅ Anomalies detected"
else
    echo "✅ Anomalies already detected"
fi

# Show summary
echo ""
echo "📈 Project Summary:"
echo "=================="
echo "📊 Data: $(wc -l < data/sensor_data.csv) sensor readings"
echo "🤖 Model: Isolation Forest trained"
echo "🚨 Anomalies: $(python -c "import pandas as pd; df=pd.read_csv('data/sensor_data_with_anomalies.csv'); print(df['anomaly'].sum())") detected"
echo ""

# Ask user if they want to start dashboard
read -p "🎯 Start dashboard now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting dashboard..."
    echo "📱 Open browser to: http://localhost:8501"
    echo "⏹️  Press Ctrl+C to stop"
    echo ""
    streamlit run dashboard.py
else
    echo "✅ Setup complete! Run 'streamlit run dashboard.py' to start dashboard"
fi
