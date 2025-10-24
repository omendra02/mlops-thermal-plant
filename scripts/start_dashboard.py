#!/usr/bin/env python3
"""
Start Dashboard Script for Thermal Plant MLOps
"""

import os
import sys
import subprocess
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Start the Streamlit dashboard"""
    parser = argparse.ArgumentParser(description="Start Thermal Plant MLOps Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port for the dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the dashboard")
    parser.add_argument("--config", default="config", help="Configuration directory")
    
    args = parser.parse_args()
    
    # Check if models exist
    model_files = [
        "models/isolation_forest.pkl",
        "models/lstm_autoencoder.h5",
        "models/ensemble_metadata.pkl"
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print("⚠️  Warning: Some models are missing:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nRun 'python scripts/train_models.py' to train models first.")
        print("Dashboard will start with limited functionality.\n")
    
    # Check if data exists
    if not os.path.exists("data/sensor_data.csv"):
        print("⚠️  Warning: Sensor data not found.")
        print("Run 'python data/generate_data.py' to generate sample data first.\n")
    
    # Start Streamlit dashboard
    dashboard_path = "mlops_thermal_plant/dashboard/dashboard_app.py"
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    print(f"🚀 Starting Thermal Plant MLOps Dashboard...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Config: {args.config}")
    print(f"\n📊 Dashboard will be available at: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop the dashboard\n")
    
    try:
        # Start Streamlit
        cmd = [
            "streamlit", "run", dashboard_path,
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
