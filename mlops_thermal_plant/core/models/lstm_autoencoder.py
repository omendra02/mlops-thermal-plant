"""
LSTM Autoencoder for Advanced Anomaly Detection in Thermal Power Plants
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
import joblib
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class LSTMAutoencoder:
    """
    LSTM Autoencoder for time-series anomaly detection in thermal power plants
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM Autoencoder
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        
        # Model architecture parameters
        self.sequence_length = config.get("sequence_length", 60)
        self.n_features = config.get("features", 8)
        self.encoding_dim = config.get("encoding_dim", 32)
        self.hidden_dims = config.get("hidden_dims", [64, 32])
        self.dropout_rate = config.get("dropout_rate", 0.2)
        
        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.optimizer = config.get("optimizer", "adam")
        self.loss_function = config.get("loss_function", "mse")
        
        # Anomaly detection parameters
        self.threshold_percentile = config.get("threshold_percentile", 95)
        self.reconstruction_threshold = config.get("reconstruction_threshold")
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = None
        self.reconstruction_errors = None
        
        # Feature names
        self.feature_names = [
            "steam_temperature", "steam_pressure", "turbine_vibration",
            "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
            "oxygen_level", "load_factor"
        ]
        
    def _build_model(self) -> Model:
        """
        Build LSTM Autoencoder architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        encoder = LSTM(self.hidden_dims[0], return_sequences=True, dropout=self.dropout_rate)(inputs)
        encoder = LSTM(self.hidden_dims[1], return_sequences=False, dropout=self.dropout_rate)(encoder)
        encoder = Dropout(self.dropout_rate)(encoder)
        
        # Bottleneck (encoding)
        encoded = Dense(self.encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = RepeatVector(self.sequence_length)(encoded)
        decoder = LSTM(self.hidden_dims[1], return_sequences=True, dropout=self.dropout_rate)(decoder)
        decoder = LSTM(self.hidden_dims[0], return_sequences=True, dropout=self.dropout_rate)(decoder)
        decoder = Dropout(self.dropout_rate)(decoder)
        
        # Output layer
        outputs = TimeDistributed(Dense(self.n_features))(decoder)
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        if self.optimizer == "adam":
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            optimizer = self.optimizer
            
        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['mae'])
        
        return model
        
    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare sequences for LSTM input
        
        Args:
            data: Input data array (n_samples, n_features)
            
        Returns:
            Sequences array (n_sequences, sequence_length, n_features)
        """
        sequences = []
        
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
            
        return np.array(sequences)
        
    def _calculate_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for input data
        
        Args:
            X: Input sequences (n_sequences, sequence_length, n_features)
            
        Returns:
            Reconstruction errors (n_sequences,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating reconstruction errors")
            
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction errors (MSE for each sequence)
        reconstruction_errors = np.mean(np.square(X - predictions), axis=(1, 2))
        
        return reconstruction_errors
        
    def fit(self, X: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the LSTM Autoencoder
        
        Args:
            X: Training data DataFrame with sensor readings
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting LSTM Autoencoder training...")
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            # Select only sensor features
            feature_data = X[self.feature_names].values
        else:
            feature_data = X
            
        # Normalize data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        
        # Split data
        split_idx = int(len(sequences) * (1 - validation_split))
        X_train = sequences[:split_idx]
        X_val = sequences[split_idx:]
        
        # Build model
        self.model = self._build_model()
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train model
        logger.info(f"Training on {len(X_train)} sequences, validating on {len(X_val)} sequences")
        
        history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = target
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        self.is_trained = True
        
        # Calculate reconstruction threshold
        train_errors = self._calculate_reconstruction_errors(X_train)
        self.reconstruction_threshold = np.percentile(train_errors, self.threshold_percentile)
        
        logger.info(f"Training completed. Reconstruction threshold: {self.reconstruction_threshold:.4f}")
        
        return self.training_history
        
    def _setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping_config = self.config.get("early_stopping", {})
        if early_stopping_config.get("patience", 10) > 0:
            early_stopping = EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=early_stopping_config.get("patience", 10),
                restore_best_weights=early_stopping_config.get("restore_best_weights", True),
                verbose=1
            )
            callbacks.append(early_stopping)
            
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'models/lstm_autoencoder_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
        
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
            
        # Prepare data
        if isinstance(X, pd.DataFrame):
            feature_data = X[self.feature_names].values
        else:
            feature_data = X
            
        # Normalize data
        scaled_data = self.scaler.transform(feature_data)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        
        # Calculate reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_errors(sequences)
        
        # Predict anomalies
        predictions = (reconstruction_errors > self.reconstruction_threshold).astype(int)
        
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probabilities
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Anomaly probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Prepare data
        if isinstance(X, pd.DataFrame):
            feature_data = X[self.feature_names].values
        else:
            feature_data = X
            
        # Normalize data
        scaled_data = self.scaler.transform(feature_data)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        
        # Calculate reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_errors(sequences)
        
        # Convert to probabilities (normalize by threshold)
        probabilities = np.clip(reconstruction_errors / self.reconstruction_threshold, 0, 1)
        
        return probabilities
        
    def get_reconstruction_errors(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get reconstruction errors for input data
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Reconstruction errors
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating reconstruction errors")
            
        # Prepare data
        if isinstance(X, pd.DataFrame):
            feature_data = X[self.feature_names].values
        else:
            feature_data = X
            
        # Normalize data
        scaled_data = self.scaler.transform(feature_data)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        
        # Calculate reconstruction errors
        reconstruction_errors = self._calculate_reconstruction_errors(sequences)
        
        return reconstruction_errors
        
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
        
        metrics = {
            "reconstruction_threshold": self.reconstruction_threshold,
            "anomaly_rate": np.mean(predictions),
            "mean_anomaly_probability": np.mean(probabilities)
        }
        
        # If true labels are provided, calculate classification metrics
        if y is not None:
            # Align labels with predictions (labels might be shorter due to sequence creation)
            if len(y) > len(predictions):
                y_aligned = y[-len(predictions):]
            else:
                y_aligned = y
                
            metrics.update({
                "accuracy": np.mean(predictions == y_aligned),
                "precision": classification_report(y_aligned, predictions, output_dict=True)["1"]["precision"],
                "recall": classification_report(y_aligned, predictions, output_dict=True)["1"]["recall"],
                "f1_score": classification_report(y_aligned, predictions, output_dict=True)["1"]["f1-score"]
            })
            
        return metrics
        
    def save_model(self, filepath: str):
        """
        Save trained model and scaler
        
        Args:
            filepath: Base filepath for saving (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Save Keras model
        model_path = f"{filepath}.h5"
        self.model.save(model_path)
        
        # Save scaler and metadata
        metadata = {
            "scaler": self.scaler,
            "reconstruction_threshold": self.reconstruction_threshold,
            "config": self.config,
            "feature_names": self.feature_names,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "training_history": self.training_history
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
        # Load Keras model
        model_path = f"{filepath}.h5"
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.scaler = metadata["scaler"]
        self.reconstruction_threshold = metadata["reconstruction_threshold"]
        self.config = metadata["config"]
        self.feature_names = metadata["feature_names"]
        self.sequence_length = metadata["sequence_length"]
        self.n_features = metadata["n_features"]
        self.training_history = metadata["training_history"]
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
        
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
            
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary
        
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance based on reconstruction errors
        
        Args:
            X: Input data DataFrame
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating feature importance")
            
        # Prepare data
        feature_data = X[self.feature_names].values
        scaled_data = self.scaler.transform(feature_data)
        sequences = self._prepare_sequences(scaled_data)
        
        # Get predictions
        predictions = self.model.predict(sequences, verbose=0)
        
        # Calculate per-feature reconstruction errors
        feature_errors = np.mean(np.square(sequences - predictions), axis=1)  # (n_sequences, n_features)
        
        # Calculate importance as mean error per feature
        feature_importance = np.mean(feature_errors, axis=0)
        
        # Normalize to sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return dict(zip(self.feature_names, feature_importance))
