"""
Data Processing Module for Thermal Plant MLOps
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing and feature engineering for thermal plant data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor
        
        Args:
            config: Data processing configuration
        """
        self.config = config
        
        # Preprocessing configuration
        self.missing_value_strategy = config.get("missing_values", {}).get("strategy", "interpolate")
        self.outlier_method = config.get("outlier_detection", {}).get("method", "iqr")
        self.outlier_threshold = config.get("outlier_detection", {}).get("threshold", 3.0)
        self.scaling_method = config.get("scaling", {}).get("method", "standard")
        self.scaling_features = config.get("scaling", {}).get("features", [])
        
        # Feature engineering configuration
        self.feature_config = config.get("feature_engineering", {})
        
        # Initialize scalers
        self.scalers = {}
        self.feature_selectors = {}
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various sources
        
        Args:
            filepath: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath, **kwargs)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            logger.info(f"Loaded data from {filepath}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, filepath: str, **kwargs):
        """
        Save data to various formats
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            **kwargs: Additional arguments for pandas write functions
        """
        try:
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False, **kwargs)
            elif filepath.endswith('.parquet'):
                df.to_parquet(filepath, index=False, **kwargs)
            elif filepath.endswith('.json'):
                df.to_json(filepath, orient='records', **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            logger.info(f"Saved data to {filepath}: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            raise
            
    def validate_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against schema
        
        Args:
            df: DataFrame to validate
            schema: Data validation schema
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = schema.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            
        # Check data types
        data_types = schema.get("data_types", {})
        for field, expected_type in data_types.items():
            if field in df.columns:
                actual_type = str(df[field].dtype)
                if expected_type not in actual_type:
                    errors.append(f"Field '{field}' has type {actual_type}, expected {expected_type}")
                    
        # Check data quality
        quality_checks = schema.get("quality_checks", {})
        
        # Completeness check
        completeness_threshold = quality_checks.get("completeness", 0.95)
        for col in df.columns:
            completeness = 1 - df[col].isnull().sum() / len(df)
            if completeness < completeness_threshold:
                errors.append(f"Column '{col}' completeness {completeness:.2%} below threshold {completeness_threshold:.2%}")
                
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        if self.missing_value_strategy == "drop":
            df_processed = df_processed.dropna()
            
        elif self.missing_value_strategy == "fill_mean":
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
                df_processed[numeric_columns].mean()
            )
            
        elif self.missing_value_strategy == "fill_median":
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
                df_processed[numeric_columns].median()
            )
            
        elif self.missing_value_strategy == "interpolate":
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].interpolate(method='linear')
            
        # Fill remaining missing values with forward fill, then backward fill
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Handled missing values using strategy: {self.missing_value_strategy}")
        return df_processed
        
    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers in the dataset
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (default: all numeric columns)
            
        Returns:
            DataFrame with outlier flags
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns
            
        outlier_columns = []
        
        for col in columns:
            if col in df_processed.columns:
                if self.outlier_method == "iqr":
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                elif self.outlier_method == "zscore":
                    mean = df_processed[col].mean()
                    std = df_processed[col].std()
                    lower_bound = mean - self.outlier_threshold * std
                    upper_bound = mean + self.outlier_threshold * std
                    
                else:
                    continue
                    
                outlier_flag = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                outlier_col_name = f"{col}_outlier"
                df_processed[outlier_col_name] = outlier_flag.astype(int)
                outlier_columns.append(outlier_col_name)
                
        logger.info(f"Detected outliers in {len(outlier_columns)} columns")
        return df_processed
        
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with time features
        """
        df_processed = df.copy()
        
        if timestamp_col not in df_processed.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df_processed
            
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_processed[timestamp_col]):
            df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col])
            
        time_config = self.feature_config.get("time_features", {})
        
        if time_config.get("hour_of_day", True):
            df_processed["hour_of_day"] = df_processed[timestamp_col].dt.hour
            
        if time_config.get("day_of_week", True):
            df_processed["day_of_week"] = df_processed[timestamp_col].dt.dayofweek
            
        if time_config.get("month", True):
            df_processed["month"] = df_processed[timestamp_col].dt.month
            
        if time_config.get("is_weekend", True):
            df_processed["is_weekend"] = df_processed[timestamp_col].dt.weekday >= 5
            
        logger.info("Created time-based features")
        return df_processed
        
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create rolling statistical features
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            
        Returns:
            DataFrame with rolling features
        """
        df_processed = df.copy()
        
        rolling_windows = self.feature_config.get("rolling_windows", [5, 10, 30, 60])
        rolling_stats = self.feature_config.get("rolling_stats", ["mean", "std", "min", "max"])
        
        for col in columns:
            if col in df_processed.columns:
                for window in rolling_windows:
                    for stat in rolling_stats:
                        if stat == "mean":
                            df_processed[f"{col}_rolling_{stat}_{window}"] = df_processed[col].rolling(window=window, min_periods=1).mean()
                        elif stat == "std":
                            df_processed[f"{col}_rolling_{stat}_{window}"] = df_processed[col].rolling(window=window, min_periods=1).std()
                        elif stat == "min":
                            df_processed[f"{col}_rolling_{stat}_{window}"] = df_processed[col].rolling(window=window, min_periods=1).min()
                        elif stat == "max":
                            df_processed[f"{col}_rolling_{stat}_{window}"] = df_processed[col].rolling(window=window, min_periods=1).max()
                            
        logger.info(f"Created rolling features for {len(columns)} columns")
        return df_processed
        
    def create_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: Input DataFrame
            columns: Columns to create lag features for
            
        Returns:
            DataFrame with lag features
        """
        df_processed = df.copy()
        
        lag_features = self.feature_config.get("lag_features", [1, 2, 3, 5, 10])
        
        for col in columns:
            if col in df_processed.columns:
                for lag in lag_features:
                    df_processed[f"{col}_lag_{lag}"] = df_processed[col].shift(lag)
                    
        logger.info(f"Created lag features for {len(columns)} columns")
        return df_processed
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for thermal plant data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        df_processed = df.copy()
        
        derived_config = self.feature_config.get("derived_features", {})
        
        # Heat rate calculation
        if derived_config.get("heat_rate", True):
            if "steam_temperature" in df_processed.columns and "steam_pressure" in df_processed.columns:
                # Simplified heat rate calculation
                df_processed["heat_rate"] = (
                    10000 - 3 * df_processed["steam_temperature"] + 
                    2 * df_processed["steam_pressure"]
                )
                
        # Efficiency calculation
        if derived_config.get("efficiency", True):
            if "heat_rate" in df_processed.columns:
                df_processed["efficiency"] = np.clip(
                    45 - (df_processed["heat_rate"] - 8500) / 100, 0, 100
                )
                
        # Equipment health score
        if derived_config.get("equipment_health", True):
            health_columns = [
                "steam_temperature", "steam_pressure", "turbine_vibration",
                "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
                "oxygen_level", "load_factor"
            ]
            
            available_health_columns = [col for col in health_columns if col in df_processed.columns]
            
            if available_health_columns:
                # Simple health score calculation (normalize and average)
                health_scores = []
                for col in available_health_columns:
                    col_data = df_processed[col].dropna()
                    if len(col_data) > 0:
                        # Normalize to 0-1 scale (assuming normal ranges)
                        normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 1e-8)
                        health_scores.append(normalized)
                        
                if health_scores:
                    df_processed["equipment_health"] = np.mean(health_scores, axis=0)
                    
        logger.info("Created derived features")
        return df_processed
        
    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (default: use scaling_features from config)
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = self.scaling_features
            
        # Filter to available columns
        columns = [col for col in columns if col in df_processed.columns]
        
        if not columns:
            logger.warning("No columns found for scaling")
            return df_processed
            
        # Initialize scaler if not exists
        scaler_key = self.scaling_method
        if scaler_key not in self.scalers:
            if self.scaling_method == "standard":
                self.scalers[scaler_key] = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scalers[scaler_key] = MinMaxScaler()
            elif self.scaling_method == "robust":
                self.scalers[scaler_key] = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}")
                return df_processed
                
        scaler = self.scalers[scaler_key]
        
        # Scale features
        if fit:
            scaled_data = scaler.fit_transform(df_processed[columns])
        else:
            scaled_data = scaler.transform(df_processed[columns])
            
        # Create new column names for scaled features
        scaled_columns = [f"{col}_scaled" for col in columns]
        df_processed[scaled_columns] = scaled_data
        
        logger.info(f"Scaled {len(columns)} features using {self.scaling_method} method")
        return df_processed
        
    def select_features(self, X: pd.DataFrame, y: np.ndarray, k: int = 10, 
                       method: str = "mutual_info") -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select top k features
        
        Args:
            X: Feature DataFrame
            y: Target array
            k: Number of features to select
            method: Feature selection method
            
        Returns:
            Tuple of (selected_features, feature_scores)
        """
        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        X_selected = selector.fit_transform(X, y)
        feature_scores = selector.scores_
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store selector for later use
        self.feature_selectors[method] = selector
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=selected_features), feature_scores
        
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = "anomaly",
                            test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Create time features
        df_processed = self.create_time_features(df_processed)
        
        # Create rolling features for sensor columns
        sensor_columns = [
            "steam_temperature", "steam_pressure", "turbine_vibration",
            "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
            "oxygen_level", "load_factor"
        ]
        available_sensor_columns = [col for col in sensor_columns if col in df_processed.columns]
        
        if available_sensor_columns:
            df_processed = self.create_rolling_features(df_processed, available_sensor_columns)
            
        # Create derived features
        df_processed = self.create_derived_features(df_processed)
        
        # Separate features and target
        feature_columns = [col for col in df_processed.columns 
                          if col not in ["timestamp", "plant_name", "fuel_type", target_column]]
        
        X = df_processed[feature_columns]
        y = df_processed[target_column] if target_column in df_processed.columns else None
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            y_train, y_test = None, None
            
        logger.info(f"Prepared training data: X_train {X_train.shape}, X_test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Add base sensor features
        base_features = [
            "steam_temperature", "steam_pressure", "turbine_vibration",
            "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
            "oxygen_level", "load_factor"
        ]
        feature_names.extend(base_features)
        
        # Add time features
        time_features = ["hour_of_day", "day_of_week", "month", "is_weekend"]
        feature_names.extend(time_features)
        
        # Add derived features
        derived_features = ["heat_rate", "efficiency", "equipment_health"]
        feature_names.extend(derived_features)
        
        return feature_names
        
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor components
        
        Args:
            filepath: Base filepath for saving
        """
        import joblib
        
        preprocessor_data = {
            "scalers": self.scalers,
            "feature_selectors": self.feature_selectors,
            "config": self.config
        }
        
        joblib.dump(preprocessor_data, f"{filepath}_preprocessor.pkl")
        logger.info(f"Saved preprocessor to {filepath}_preprocessor.pkl")
        
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor components
        
        Args:
            filepath: Base filepath for loading
        """
        import joblib
        
        preprocessor_data = joblib.load(f"{filepath}_preprocessor.pkl")
        
        self.scalers = preprocessor_data["scalers"]
        self.feature_selectors = preprocessor_data["feature_selectors"]
        self.config = preprocessor_data["config"]
        
        logger.info(f"Loaded preprocessor from {filepath}_preprocessor.pkl")
