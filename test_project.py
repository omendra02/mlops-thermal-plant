#!/usr/bin/env python3
"""
Comprehensive Test Script for Thermal Plant MLOps Project
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import subprocess
import time

def print_status(message, status="INFO"):
    """Print status message with emoji"""
    emoji_map = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "ERROR": "❌",
        "WARNING": "⚠️",
        "TEST": "🧪"
    }
    print(f"{emoji_map.get(status, 'ℹ️')} {message}")

def test_data_generation():
    """Test data generation functionality"""
    print_status("Testing data generation...", "TEST")
    
    try:
        # Check if data file exists
        if not os.path.exists("data/sensor_data.csv"):
            print_status("Data file not found, generating...", "WARNING")
            result = subprocess.run([sys.executable, "data/generate_data.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print_status(f"Data generation failed: {result.stderr}", "ERROR")
                return False
        
        # Load and validate data
        df = pd.read_csv("data/sensor_data.csv")
        
        required_columns = ["timestamp", "plant_name", "fuel_type", "capacity_mw",
                           "temperature", "vibration", "pressure", "flow_rate", "load_factor"]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print_status(f"Missing columns: {missing_cols}", "ERROR")
            return False
            
        if len(df) < 100:
            print_status(f"Too few data points: {len(df)}", "ERROR")
            return False
            
        print_status(f"Data generation test passed: {len(df)} rows, {len(df.columns)} columns", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Data generation test failed: {e}", "ERROR")
        return False

def test_model_training():
    """Test model training functionality"""
    print_status("Testing model training...", "TEST")
    
    try:
        # Check if model file exists, if not train it
        if not os.path.exists("model/isolation_forest.pkl"):
            print_status("Model not found, training...", "WARNING")
            result = subprocess.run([sys.executable, "src/train.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print_status(f"Model training failed: {result.stderr}", "ERROR")
                return False
        
        # Load and validate model
        model = joblib.load("model/isolation_forest.pkl")
        
        if not hasattr(model, 'predict'):
            print_status("Model doesn't have predict method", "ERROR")
            return False
            
        print_status("Model training test passed", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Model training test failed: {e}", "ERROR")
        return False

def test_anomaly_prediction():
    """Test anomaly prediction functionality"""
    print_status("Testing anomaly prediction...", "TEST")
    
    try:
        # Check if prediction file exists, if not run prediction
        if not os.path.exists("data/sensor_data_with_anomalies.csv"):
            print_status("Prediction file not found, running prediction...", "WARNING")
            result = subprocess.run([sys.executable, "src/predict.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print_status(f"Prediction failed: {result.stderr}", "ERROR")
                return False
        
        # Load and validate predictions
        df = pd.read_csv("data/sensor_data_with_anomalies.csv")
        
        if "anomaly" not in df.columns:
            print_status("Anomaly column not found in predictions", "ERROR")
            return False
            
        anomaly_count = df["anomaly"].sum()
        total_count = len(df)
        anomaly_rate = anomaly_count / total_count
        
        print_status(f"Anomaly prediction test passed: {anomaly_count}/{total_count} anomalies ({anomaly_rate:.2%})", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Anomaly prediction test failed: {e}", "ERROR")
        return False

def test_dashboard_import():
    """Test dashboard import functionality"""
    print_status("Testing dashboard import...", "TEST")
    
    try:
        # Test basic dashboard import
        import streamlit as st
        print_status(f"Streamlit version: {st.__version__}", "SUCCESS")
        
        # Test if dashboard code can be imported
        with open("dashboard.py", "r") as f:
            dashboard_code = f.read()
        
        # Basic syntax check
        compile(dashboard_code, "dashboard.py", "exec")
        print_status("Dashboard code syntax is valid", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Dashboard import test failed: {e}", "ERROR")
        return False

def test_dependencies():
    """Test required dependencies"""
    print_status("Testing dependencies...", "TEST")
    
    required_packages = [
        "pandas", "numpy", "sklearn", "joblib", "streamlit"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package}: Available", "SUCCESS")
        except ImportError:
            missing_packages.append(package)
            print_status(f"{package}: Missing", "ERROR")
    
    if missing_packages:
        print_status(f"Missing packages: {missing_packages}", "ERROR")
        print_status("Install with: pip install " + " ".join(missing_packages), "INFO")
        return False
    
    return True

def test_file_structure():
    """Test project file structure"""
    print_status("Testing file structure...", "TEST")
    
    required_files = [
        "data/sensor_data.csv",
        "model/isolation_forest.pkl", 
        "data/sensor_data_with_anomalies.csv",
        "src/train.py",
        "src/predict.py",
        "dashboard.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_status(f"{file_path}: Found", "SUCCESS")
        else:
            missing_files.append(file_path)
            print_status(f"{file_path}: Missing", "ERROR")
    
    if missing_files:
        print_status(f"Missing files: {missing_files}", "ERROR")
        return False
    
    return True

def test_advanced_modules():
    """Test advanced modules availability"""
    print_status("Testing advanced modules...", "TEST")
    
    advanced_packages = [
        "tensorflow", "mlflow", "paho.mqtt", "plotly", "yaml"
    ]
    
    available_advanced = []
    missing_advanced = []
    
    for package in advanced_packages:
        try:
            __import__(package)
            available_advanced.append(package)
            print_status(f"{package}: Available", "SUCCESS")
        except ImportError:
            missing_advanced.append(package)
            print_status(f"{package}: Missing", "WARNING")
    
    if missing_advanced:
        print_status(f"Advanced packages missing: {missing_advanced}", "WARNING")
        print_status("Install with: pip install " + " ".join(missing_advanced), "INFO")
        return False
    
    return True

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Project Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("Data Generation", test_data_generation),
        ("Model Training", test_model_training),
        ("Anomaly Prediction", test_anomaly_prediction),
        ("Dashboard Import", test_dashboard_import),
        ("Advanced Modules", test_advanced_modules),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Test {test_name} crashed: {e}", "ERROR")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print_status("🎉 All tests passed! Project is working correctly.", "SUCCESS")
    elif passed >= 5:  # Basic functionality working
        print_status("✅ Basic functionality is working! Advanced features need setup.", "SUCCESS")
    else:
        print_status("⚠️ Some basic tests failed. Check the errors above.", "WARNING")
    
    return results

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test - only basic functionality
        print("🧪 Running Quick Test...")
        test_dependencies()
        test_file_structure()
        test_data_generation()
        test_model_training()
        test_anomaly_prediction()
    else:
        # Full comprehensive test
        run_comprehensive_test()

if __name__ == "__main__":
    main()
