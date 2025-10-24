#!/usr/bin/env python3
"""
Model Training Script for Thermal Plant MLOps
"""

import os
import sys
import yaml
import logging
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops_thermal_plant.core.models import LSTMAutoencoder, IsolationForestModel, EnsembleAnomalyDetector
from mlops_thermal_plant.core.data import MLflowManager, DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config"):
    """Load configuration files"""
    config = {}
    
    config_files = [
        "model_config.yaml",
        "plant_config.yaml", 
        "database_config.yaml"
    ]
    
    for config_file in config_files:
        file_path = os.path.join(config_path, config_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config_name = config_file.replace('.yaml', '').replace('_config', '')
                config[config_name] = yaml.safe_load(f)
        else:
            logger.warning(f"Configuration file not found: {file_path}")
            
    return config


def load_data(data_path: str = "data/sensor_data.csv"):
    """Load training data"""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run data/generate_data.py first")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")
    
    return df


def train_models(config: dict, data: pd.DataFrame):
    """Train all models"""
    
    # Initialize MLflow manager
    mlflow_config = {
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment_name": "thermal_plant_anomaly_detection"
    }
    mlflow_manager = MLflowManager(mlflow_config)
    
    # Initialize data processor
    data_processor = DataProcessor(config.get("model", {}).get("preprocessing", {}))
    
    # Prepare training data
    X_train, X_test, y_train, y_test = data_processor.prepare_training_data(
        data, target_column="anomaly"
    )
    
    models = {}
    training_results = {}
    
    # Train Isolation Forest
    logger.info("Training Isolation Forest model...")
    with mlflow_manager.start_run(run_name="isolation_forest_training"):
        if_model = IsolationForestModel(config["model"]["models"]["isolation_forest"])
        if_result = if_model.fit(X_train, y_train)
        
        # Evaluate model
        if_metrics = if_model.evaluate(X_test, y_test)
        
        # Log metrics
        mlflow_manager.log_metrics(if_metrics)
        mlflow_manager.log_parameters(config["model"]["models"]["isolation_forest"])
        
        # Save model
        if_model.save_model("models/isolation_forest")
        models["isolation_forest"] = if_model
        training_results["isolation_forest"] = if_result
        
        logger.info(f"Isolation Forest training completed: {if_metrics}")
    
    # Train LSTM Autoencoder
    logger.info("Training LSTM Autoencoder model...")
    with mlflow_manager.start_run(run_name="lstm_autoencoder_training"):
        lstm_model = LSTMAutoencoder(config["model"]["models"]["lstm_autoencoder"])
        lstm_result = lstm_model.fit(X_train)
        
        # Evaluate model
        lstm_metrics = lstm_model.evaluate(X_test, y_test)
        
        # Log metrics
        mlflow_manager.log_metrics(lstm_metrics)
        mlflow_manager.log_parameters(config["model"]["models"]["lstm_autoencoder"])
        
        # Save model
        lstm_model.save_model("models/lstm_autoencoder")
        models["lstm_autoencoder"] = lstm_model
        training_results["lstm_autoencoder"] = lstm_result
        
        logger.info(f"LSTM Autoencoder training completed: {lstm_metrics}")
    
    # Train Ensemble Model
    if len(models) >= 2:
        logger.info("Training Ensemble model...")
        with mlflow_manager.start_run(run_name="ensemble_training"):
            # Update ensemble config with trained models
            ensemble_config = config["model"]["models"]["ensemble"].copy()
            ensemble_config["model_configs"] = {
                "isolation_forest": config["model"]["models"]["isolation_forest"],
                "lstm_autoencoder": config["model"]["models"]["lstm_autoencoder"]
            }
            
            ensemble_model = EnsembleAnomalyDetector(ensemble_config)
            ensemble_result = ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_metrics = ensemble_model.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow_manager.log_metrics(ensemble_metrics)
            mlflow_manager.log_parameters(ensemble_config)
            
            # Save model
            ensemble_model.save_models("models/ensemble")
            models["ensemble"] = ensemble_model
            training_results["ensemble"] = ensemble_result
            
            logger.info(f"Ensemble training completed: {ensemble_metrics}")
    
    return models, training_results


def main():
    """Main training function"""
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    data = load_data()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train models
    models, results = train_models(config, data)
    
    logger.info("Model training pipeline completed successfully!")
    logger.info(f"Trained {len(models)} models")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
    
    print("\nModels saved to 'models/' directory")
    print("MLflow experiments available at http://localhost:5001")


if __name__ == "__main__":
    main()
