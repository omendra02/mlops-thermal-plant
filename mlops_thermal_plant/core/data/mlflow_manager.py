"""
MLflow Manager for Experiment Tracking and Model Registry
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import yaml

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    MLflow manager for experiment tracking and model registry
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLflow manager
        
        Args:
            config: MLflow configuration dictionary
        """
        self.config = config
        
        # MLflow settings
        self.tracking_uri = config.get("tracking_uri", "sqlite:///mlflow.db")
        self.registry_uri = config.get("registry_uri", self.tracking_uri)
        self.experiment_name = config.get("experiment_name", "thermal_plant_anomaly_detection")
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Get or create experiment
        self.experiment_id = self._get_or_create_experiment()
        
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        # Set environment variables for MLflow
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        
    def _get_or_create_experiment(self) -> str:
        """
        Get or create MLflow experiment
        
        Returns:
            Experiment ID
        """
        try:
            # Try to get existing experiment
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                logger.info(f"Using existing experiment: {self.experiment_name}")
                return experiment.experiment_id
                
        except MlflowException:
            pass
            
        # Create new experiment
        experiment_id = self.client.create_experiment(self.experiment_name)
        logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
        
        return experiment_id
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Active MLflow run
        """
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags or {}
        ) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run
            
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to current run
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to current run
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
        
    def log_artifacts(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """
        Log artifacts to current run
        
        Args:
            artifact_path: Path to the artifact file or directory
            artifact_dir: Directory name in MLflow UI (optional)
        """
        mlflow.log_artifacts(artifact_path, artifact_dir)
        logger.debug(f"Logged artifacts from: {artifact_path}")
        
    def log_model(self, model, model_name: str, signature=None, input_example=None, 
                  registered_model_name: Optional[str] = None) -> str:
        """
        Log model to MLflow
        
        Args:
            model: Model object to log
            model_name: Name for the model
            signature: Model signature (optional)
            input_example: Example input (optional)
            registered_model_name: Name for registered model (optional)
            
        Returns:
            Model URI
        """
        try:
            # Determine model flavor
            model_type = type(model).__name__.lower()
            
            if "lstm" in model_type or "tensorflow" in str(type(model)):
                # TensorFlow/Keras model
                model_uri = mlflow.tensorflow.log_model(
                    model.model,  # Keras model
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example
                )
                
            elif "isolation" in model_type or "sklearn" in str(type(model)):
                # Scikit-learn model
                model_uri = mlflow.sklearn.log_model(
                    model.model,  # sklearn model
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example
                )
                
            else:
                # Generic model
                model_uri = mlflow.pyfunc.log_model(
                    artifact_path=model_name,
                    python_model=model,
                    signature=signature,
                    input_example=input_example
                )
                
            logger.info(f"Logged model: {model_name} (URI: {model_uri})")
            
            # Register model if name provided
            if registered_model_name:
                self.register_model(model_uri, registered_model_name)
                
            return model_uri
            
        except Exception as e:
            logger.error(f"Error logging model {model_name}: {e}")
            raise
            
    def register_model(self, model_uri: str, name: str, version: Optional[str] = None) -> str:
        """
        Register model in MLflow model registry
        
        Args:
            model_uri: URI of the model to register
            name: Name for the registered model
            version: Version for the model (optional)
            
        Returns:
            Registered model version
        """
        try:
            result = mlflow.register_model(model_uri, name)
            logger.info(f"Registered model: {name} (Version: {result.version})")
            return result.version
            
        except MlflowException as e:
            logger.error(f"Error registering model {name}: {e}")
            raise
            
    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get list of registered models
        
        Returns:
            List of registered model information
        """
        try:
            models = self.client.search_registered_models()
            model_info = []
            
            for model in models:
                model_info.append({
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "description": v.description,
                            "creation_timestamp": v.creation_timestamp
                        }
                        for v in model.latest_versions
                    ]
                })
                
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting registered models: {e}")
            return []
            
    def promote_model(self, name: str, version: str, stage: str):
        """
        Promote model to a specific stage
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage (Staging, Production, etc.)
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted model {name} version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error promoting model {name}: {e}")
            raise
            
    def get_latest_model(self, name: str, stage: str = "Production") -> Optional[str]:
        """
        Get latest model version for a specific stage
        
        Args:
            name: Model name
            stage: Model stage
            
        Returns:
            Model URI or None
        """
        try:
            model_versions = self.client.get_latest_versions(name, stages=[stage])
            
            if model_versions:
                latest_version = model_versions[0]
                return f"models:/{name}/{latest_version.version}"
                
        except Exception as e:
            logger.error(f"Error getting latest model {name}: {e}")
            
        return None
        
    def load_model(self, model_uri: str) -> Any:
        """
        Load model from MLflow
        
        Args:
            model_uri: Model URI
            
        Returns:
            Loaded model
        """
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_uri}: {e}")
            raise
            
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {"run_id": run_id}
                
                # Add parameters
                for param_name, param_value in run.data.params.items():
                    run_data[f"param_{param_name}"] = param_value
                    
                # Add metrics
                for metric_name in metrics:
                    if metric_name in run.data.metrics:
                        run_data[f"metric_{metric_name}"] = run.data.metrics[metric_name]
                    else:
                        run_data[f"metric_{metric_name}"] = None
                        
                comparison_data.append(run_data)
                
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Error comparing runs: {e}")
            return pd.DataFrame()
            
    def search_runs(self, filter_string: str = "", max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for runs with specific criteria
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results
            
        Returns:
            List of run information
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            run_info = []
            for run in runs:
                run_info.append({
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                })
                
            return run_info
            
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []
            
    def log_data_profile(self, data: pd.DataFrame, profile_name: str = "data_profile"):
        """
        Log data profile as artifact
        
        Args:
            data: DataFrame to profile
            profile_name: Name for the profile artifact
        """
        try:
            from pandas_profiling import ProfileReport
            
            # Create profile
            profile = ProfileReport(data, title=profile_name)
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                profile.to_file(f.name)
                mlflow.log_artifact(f.name, f"{profile_name}.html")
                
            # Clean up
            os.unlink(f.name)
            
            logger.info(f"Logged data profile: {profile_name}")
            
        except ImportError:
            logger.warning("pandas_profiling not available, skipping data profile")
        except Exception as e:
            logger.error(f"Error creating data profile: {e}")
            
    def log_model_signature(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Log model signature
        
        Args:
            X: Input features
            y: Target values (optional)
            
        Returns:
            Model signature
        """
        try:
            signature = infer_signature(X, y)
            mlflow.log_input(signature, context="training_data")
            return signature
            
        except Exception as e:
            logger.error(f"Error inferring model signature: {e}")
            return None
            
    def create_model_version(self, model_name: str, model_uri: str, 
                           description: str = "", tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new model version
        
        Args:
            model_name: Name of the model
            model_uri: URI of the model
            description: Description for the model version
            tags: Tags for the model version
            
        Returns:
            Model version
        """
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags or {}
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description
                )
                
            logger.info(f"Created model version: {model_name} v{result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise
            
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information
        
        Returns:
            Experiment information dictionary
        """
        try:
            experiment = self.client.get_experiment(self.experiment_id)
            
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": experiment.creation_time,
                "last_update_time": experiment.last_update_time
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment info: {e}")
            return {}
            
    def export_experiment(self, export_path: str):
        """
        Export experiment data
        
        Args:
            export_path: Path to export the experiment data
        """
        try:
            # Get all runs in the experiment
            runs = self.client.search_runs(experiment_ids=[self.experiment_id])
            
            # Export runs data
            export_data = []
            for run in runs:
                export_data.append({
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                })
                
            # Save to file
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            logger.info(f"Exported experiment data to: {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting experiment: {e}")
            raise
