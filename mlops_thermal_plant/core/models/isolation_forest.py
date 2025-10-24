"""
Enhanced Isolation Forest Model for Thermal Plant Anomaly Detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Enhanced Isolation Forest model for thermal power plant anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Isolation Forest model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        
        # Model parameters
        self.contamination = config.get("contamination", 0.05)
        self.random_state = config.get("random_state", 42)
        self.max_samples = config.get("max_samples", "auto")
        self.max_features = config.get("max_features", 1.0)
        self.bootstrap = config.get("bootstrap", False)
        self.n_estimators = config.get("n_estimators", 100)
        
        # Preprocessing
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = None
        self.is_trained = False
        
        # Initialize model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_estimators=self.n_estimators
        )
        
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare and preprocess features
        
        Args:
            X: Input DataFrame
            
        Returns:
            Preprocessed feature array
        """
        # Select sensor features
        sensor_features = [
            "steam_temperature", "steam_pressure", "turbine_vibration",
            "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
            "oxygen_level", "load_factor"
        ]
        
        # Filter available features
        available_features = [col for col in sensor_features if col in X.columns]
        
        if len(available_features) == 0:
            raise ValueError("No sensor features found in the data")
            
        # Store feature names
        if self.feature_names is None:
            self.feature_names = available_features
            
        # Extract features
        feature_data = X[available_features].values
        
        return feature_data
        
    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for better anomaly detection
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        X_enhanced = X.copy()
        
        # Efficiency-related features
        if "steam_temperature" in X.columns and "steam_pressure" in X.columns:
            X_enhanced["steam_enthalpy_approx"] = (
                X["steam_temperature"] * 4.18 + X["steam_pressure"] * 0.1
            )
            
        # Temperature ratios
        if "steam_temperature" in X.columns and "cooling_water_temp" in X.columns:
            X_enhanced["temp_ratio"] = X["steam_temperature"] / (X["cooling_water_temp"] + 1)
            
        # Vibration to load ratio
        if "turbine_vibration" in X.columns and "load_factor" in X.columns:
            X_enhanced["vibration_load_ratio"] = X["turbine_vibration"] / (X["load_factor"] + 0.1)
            
        # Pressure to flow ratio
        if "steam_pressure" in X.columns and "fuel_flow_rate" in X.columns:
            X_enhanced["pressure_flow_ratio"] = X["steam_pressure"] / (X["fuel_flow_rate"] + 0.1)
            
        # Oxygen efficiency
        if "oxygen_level" in X.columns and "fuel_flow_rate" in X.columns:
            X_enhanced["oxygen_efficiency"] = X["oxygen_level"] / (X["fuel_flow_rate"] + 0.1)
            
        # Rolling statistics (if enough data)
        if len(X) > 10:
            for col in ["steam_temperature", "steam_pressure", "turbine_vibration"]:
                if col in X.columns:
                    X_enhanced[f"{col}_rolling_mean"] = X[col].rolling(window=5, min_periods=1).mean()
                    X_enhanced[f"{col}_rolling_std"] = X[col].rolling(window=5, min_periods=1).std()
                    
        return X_enhanced
        
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the Isolation Forest model
        
        Args:
            X: Training data DataFrame
            y: Target labels (optional, for compatibility)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting Isolation Forest training...")
        
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Fit scaler and transform data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Train model
        self.model.fit(scaled_data)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(scaled_data)
        train_scores = self.model.score_samples(scaled_data)
        
        # Convert predictions (1 = normal, -1 = anomaly)
        train_anomalies = (train_predictions == -1).astype(int)
        
        training_results = {
            "n_samples": len(X),
            "n_features": feature_data.shape[1],
            "contamination": self.contamination,
            "anomaly_rate": np.mean(train_anomalies),
            "mean_anomaly_score": np.mean(train_scores),
            "feature_names": self.feature_names
        }
        
        logger.info(f"Training completed. Detected {np.sum(train_anomalies)} anomalies out of {len(X)} samples")
        
        return training_results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in the data
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Transform data
        scaled_data = self.scaler.transform(feature_data)
        
        # Predict
        predictions = self.model.predict(scaled_data)
        
        # Convert to binary (0 = normal, 1 = anomaly)
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probabilities based on decision scores
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Anomaly probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Transform data
        scaled_data = self.scaler.transform(feature_data)
        
        # Get decision scores
        scores = self.model.score_samples(scaled_data)
        
        # Convert scores to probabilities
        # Lower scores indicate higher anomaly probability
        # Normalize scores to [0, 1] range
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            probabilities = (max_score - scores) / (max_score - min_score)
        else:
            probabilities = np.zeros_like(scores)
            
        return probabilities
        
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get raw anomaly scores
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Raw anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating scores")
            
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Transform data
        scaled_data = self.scaler.transform(feature_data)
        
        # Get scores
        scores = self.model.score_samples(scaled_data)
        
        return scores
        
    def evaluate(self, X: pd.DataFrame, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X: Input data DataFrame
            y: True labels (optional)
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        # Get predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        scores = self.get_anomaly_scores(X)
        
        metrics = {
            "anomaly_rate": np.mean(predictions),
            "mean_anomaly_probability": np.mean(probabilities),
            "mean_anomaly_score": np.mean(scores),
            "score_std": np.std(scores)
        }
        
        # If true labels are provided, calculate classification metrics
        if y is not None:
            metrics.update({
                "accuracy": np.mean(predictions == y),
                "precision": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["precision"],
                "recall": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["recall"],
                "f1_score": classification_report(y, predictions, output_dict=True, zero_division=0)["1"]["f1-score"]
            })
            
        return metrics
        
    def cross_validate(self, X: pd.DataFrame, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Input data DataFrame
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Transform data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, scaled_data, cv=cv, 
            scoring='neg_mean_squared_error'
        )
        
        return {
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        }
        
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using permutation importance
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating feature importance")
            
        from sklearn.inspection import permutation_importance
        
        # Create enhanced features
        X_enhanced = self._create_derived_features(X)
        
        # Prepare features
        feature_data = self._prepare_features(X_enhanced)
        
        # Transform data
        scaled_data = self.scaler.transform(feature_data)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, scaled_data, n_repeats=10, random_state=42
        )
        
        # Create feature importance dictionary
        feature_importance = dict(zip(
            self.feature_names, 
            perm_importance.importances_mean
        ))
        
        return feature_importance
        
    def save_model(self, filepath: str):
        """
        Save trained model and scaler
        
        Args:
            filepath: Base filepath for saving (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Save model
        model_path = f"{filepath}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save scaler and metadata
        metadata = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        metadata_path = f"{filepath}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path} and metadata to {metadata_path}")
        
    def load_model(self, filepath: str):
        """
        Load trained model and scaler
        
        Args:
            filepath: Base filepath for loading (without extension)
        """
        # Load model
        model_path = f"{filepath}.pkl"
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.scaler = metadata["scaler"]
        self.feature_names = metadata["feature_names"]
        self.config = metadata["config"]
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        return {
            "model_type": "IsolationForest",
            "is_trained": self.is_trained,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0
        }
