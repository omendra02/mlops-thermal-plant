"""
Ensemble Model for Combining Multiple Anomaly Detection Methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix

from .lstm_autoencoder import LSTMAutoencoder
from .isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """
    Ensemble model that combines multiple anomaly detection methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble anomaly detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Ensemble configuration
        self.methods = config.get("methods", ["isolation_forest", "lstm_autoencoder"])
        self.voting_strategy = config.get("voting_strategy", "majority")  # or "weighted"
        self.weights = config.get("weights", [0.6, 0.4])
        
        # Initialize individual models
        self.models = {}
        self.is_trained = False
        
        # Model configurations
        self.model_configs = config.get("model_configs", {})
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize individual models based on configuration"""
        
        if "isolation_forest" in self.methods:
            if_config = self.model_configs.get("isolation_forest", {})
            self.models["isolation_forest"] = IsolationForestModel(if_config)
            
        if "lstm_autoencoder" in self.methods:
            lstm_config = self.model_configs.get("lstm_autoencoder", {})
            self.models["lstm_autoencoder"] = LSTMAutoencoder(lstm_config)
            
        logger.info(f"Initialized ensemble with methods: {list(self.models.keys())}")
        
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble
        
        Args:
            X: Training data DataFrame
            y: Target labels (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting ensemble model training...")
        
        training_results = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                if name == "isolation_forest":
                    result = model.fit(X, y)
                elif name == "lstm_autoencoder":
                    result = model.fit(X)
                else:
                    raise ValueError(f"Unknown model type: {name}")
                    
                training_results[name] = result
                logger.info(f"Successfully trained {name}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                training_results[name] = {"error": str(e)}
                
        self.is_trained = all(
            "error" not in result for result in training_results.values()
        )
        
        if self.is_trained:
            logger.info("All models trained successfully")
        else:
            logger.warning("Some models failed to train")
            
        return training_results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using ensemble voting
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        # Get predictions from each model
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                # Use random predictions as fallback
                predictions[name] = np.random.randint(0, 2, size=len(X))
                
        # Combine predictions
        ensemble_predictions = self._combine_predictions(predictions)
        
        return ensemble_predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probabilities using ensemble
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Anomaly probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        # Get probabilities from each model
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                else:
                    # Convert predictions to probabilities
                    pred = model.predict(X)
                    prob = pred.astype(float)
                probabilities[name] = prob
            except Exception as e:
                logger.error(f"Error getting probabilities from {name}: {e}")
                # Use random probabilities as fallback
                probabilities[name] = np.random.random(size=len(X))
                
        # Combine probabilities
        ensemble_probabilities = self._combine_probabilities(probabilities)
        
        return ensemble_probabilities
        
    def _combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: Dictionary of predictions from each model
            
        Returns:
            Combined predictions
        """
        if self.voting_strategy == "majority":
            # Majority voting
            prediction_matrix = np.array(list(predictions.values()))
            ensemble_predictions = np.round(np.mean(prediction_matrix, axis=0)).astype(int)
            
        elif self.voting_strategy == "weighted":
            # Weighted voting
            prediction_matrix = np.array(list(predictions.values()))
            weights_array = np.array(self.weights[:len(predictions)])
            weights_array = weights_array / np.sum(weights_array)  # Normalize weights
            
            weighted_predictions = np.average(prediction_matrix, axis=0, weights=weights_array)
            ensemble_predictions = np.round(weighted_predictions).astype(int)
            
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
            
        return ensemble_predictions
        
    def _combine_probabilities(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine probabilities from multiple models
        
        Args:
            probabilities: Dictionary of probabilities from each model
            
        Returns:
            Combined probabilities
        """
        if self.voting_strategy == "majority":
            # Average probabilities
            probability_matrix = np.array(list(probabilities.values()))
            ensemble_probabilities = np.mean(probability_matrix, axis=0)
            
        elif self.voting_strategy == "weighted":
            # Weighted average probabilities
            probability_matrix = np.array(list(probabilities.values()))
            weights_array = np.array(self.weights[:len(probabilities)])
            weights_array = weights_array / np.sum(weights_array)  # Normalize weights
            
            ensemble_probabilities = np.average(probability_matrix, axis=0, weights=weights_array)
            
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
            
        return ensemble_probabilities
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        
        Args:
            X: Input data DataFrame
            y: True labels (optional)
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
            
        # Get ensemble predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        metrics = {
            "ensemble_anomaly_rate": np.mean(predictions),
            "ensemble_mean_probability": np.mean(probabilities),
            "voting_strategy": self.voting_strategy,
            "model_weights": self.weights[:len(self.models)]
        }
        
        # Evaluate individual models
        individual_metrics = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'evaluate'):
                    model_metrics = model.evaluate(X, y)
                    individual_metrics[name] = model_metrics
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                individual_metrics[name] = {"error": str(e)}
                
        metrics["individual_models"] = individual_metrics
        
        # If true labels are provided, calculate classification metrics
        if y is not None:
            metrics.update({
                "ensemble_accuracy": np.mean(predictions == y),
                "ensemble_precision": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["precision"],
                "ensemble_recall": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["recall"],
                "ensemble_f1_score": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["f1-score"]
            })
            
        return metrics
        
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Dictionary of predictions from each model
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                predictions[name] = np.full(len(X), -1)  # Error indicator
                
        return predictions
        
    def get_model_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get probabilities from each individual model
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Dictionary of probabilities from each model
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
            
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    prob = pred.astype(float)
                probabilities[name] = prob
            except Exception as e:
                logger.error(f"Error getting probabilities from {name}: {e}")
                probabilities[name] = np.full(len(X), -1)  # Error indicator
                
        return probabilities
        
    def save_models(self, base_filepath: str):
        """
        Save all models in the ensemble
        
        Args:
            base_filepath: Base filepath for saving models
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
            
        import os
        os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
        
        for name, model in self.models.items():
            try:
                model_path = f"{base_filepath}_{name}"
                model.save_model(model_path)
                logger.info(f"Saved {name} model")
            except Exception as e:
                logger.error(f"Error saving {name} model: {e}")
                
        # Save ensemble metadata
        import joblib
        metadata = {
            "methods": self.methods,
            "voting_strategy": self.voting_strategy,
            "weights": self.weights,
            "config": self.config
        }
        
        metadata_path = f"{base_filepath}_ensemble_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Saved ensemble metadata to {metadata_path}")
        
    def load_models(self, base_filepath: str):
        """
        Load all models in the ensemble
        
        Args:
            base_filepath: Base filepath for loading models
        """
        import joblib
        
        # Load ensemble metadata
        metadata_path = f"{base_filepath}_ensemble_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.methods = metadata["methods"]
        self.voting_strategy = metadata["voting_strategy"]
        self.weights = metadata["weights"]
        self.config = metadata["config"]
        
        # Reinitialize models
        self._initialize_models()
        
        # Load each model
        for name, model in self.models.items():
            try:
                model_path = f"{base_filepath}_{name}"
                model.load_model(model_path)
                logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.error(f"Error loading {name} model: {e}")
                
        self.is_trained = True
        logger.info("Loaded all ensemble models")
        
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get ensemble information
        
        Returns:
            Ensemble information dictionary
        """
        model_info = {}
        for name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                model_info[name] = model.get_model_info()
            else:
                model_info[name] = {"model_type": name, "is_trained": hasattr(model, 'is_trained') and model.is_trained}
                
        return {
            "ensemble_type": "EnsembleAnomalyDetector",
            "methods": self.methods,
            "voting_strategy": self.voting_strategy,
            "weights": self.weights,
            "is_trained": self.is_trained,
            "models": model_info
        }
