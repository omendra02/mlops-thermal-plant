"""
Enhanced Streamlit Dashboard for Real-time Thermal Plant Monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import yaml
import os

# Import our custom modules
from ..core.models import LSTMAutoencoder, IsolationForestModel, EnsembleAnomalyDetector
from ..core.data import MLflowManager, DataProcessor
from ..iot.mqtt_client import ThermalPlantMQTTSubscriber
from ..iot.sensor_simulator import ThermalPlantSensorSimulator

logger = logging.getLogger(__name__)


class ThermalPlantDashboard:
    """
    Enhanced Streamlit dashboard for thermal power plant monitoring
    """
    
    def __init__(self, config_path: str = "config"):
        """
        Initialize dashboard
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path
        self.load_configurations()
        
        # Initialize components
        self.mlflow_manager = None
        self.data_processor = None
        self.models = {}
        self.current_data = None
        
        # Dashboard state
        self.setup_page_config()
        
    def load_configurations(self):
        """Load configuration files"""
        try:
            # Load MLflow config
            with open(f"{self.config_path}/mlflow_config.yaml", 'r') as f:
                self.mlflow_config = yaml.safe_load(f)
                
            # Load model config
            with open(f"{self.config_path}/model_config.yaml", 'r') as f:
                self.model_config = yaml.safe_load(f)
                
            # Load plant config
            with open(f"{self.config_path}/plant_config.yaml", 'r') as f:
                self.plant_config = yaml.safe_load(f)
                
            # Load MQTT config
            with open(f"{self.config_path}/mqtt_config.yaml", 'r') as f:
                self.mqtt_config = yaml.safe_load(f)
                
        except Exception as e:
            st.error(f"Error loading configurations: {e}")
            st.stop()
            
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Thermal Plant MLOps Monitor",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_components(self):
        """Initialize dashboard components"""
        try:
            # Initialize MLflow manager
            self.mlflow_manager = MLflowManager(self.mlflow_config)
            
            # Initialize data processor
            self.data_processor = DataProcessor(self.model_config.get("preprocessing", {}))
            
            # Load models
            self.load_models()
            
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            
    def load_models(self):
        """Load trained models"""
        try:
            model_dir = "models"
            
            # Load Isolation Forest model
            if os.path.exists(f"{model_dir}/isolation_forest.pkl"):
                self.models["isolation_forest"] = IsolationForestModel(self.model_config["models"]["isolation_forest"])
                self.models["isolation_forest"].load_model(f"{model_dir}/isolation_forest")
                
            # Load LSTM Autoencoder model
            if os.path.exists(f"{model_dir}/lstm_autoencoder.h5"):
                self.models["lstm_autoencoder"] = LSTMAutoencoder(self.model_config["models"]["lstm_autoencoder"])
                self.models["lstm_autoencoder"].load_model(f"{model_dir}/lstm_autoencoder")
                
            # Load Ensemble model
            if len(self.models) > 1:
                self.models["ensemble"] = EnsembleAnomalyDetector(self.model_config["models"]["ensemble"])
                self.models["ensemble"].load_models(f"{model_dir}/ensemble")
                
        except Exception as e:
            st.warning(f"Could not load models: {e}")
            
    def render_header(self):
        """Render dashboard header"""
        st.title("⚡ Thermal Plant MLOps Monitor")
        st.markdown("---")
        
        # Plant information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Plant Name", "Thermal Plant Alpha")
            
        with col2:
            st.metric("Fuel Type", "Coal")
            
        with col3:
            st.metric("Capacity", "500 MW")
            
        with col4:
            st.metric("Status", "🟢 Operational", delta="Normal")
            
    def render_sensor_overview(self, data: pd.DataFrame):
        """Render sensor overview section"""
        st.subheader("📊 Sensor Overview")
        
        # Current sensor readings
        if len(data) > 0:
            latest_data = data.iloc[-1]
            
            # Create columns for sensor metrics
            cols = st.columns(4)
            
            sensors = [
                ("steam_temperature", "🌡️ Steam Temp", "°C"),
                ("steam_pressure", "🔧 Pressure", "bar"),
                ("turbine_vibration", "⚙️ Vibration", "mm/s"),
                ("generator_temperature", "🔌 Gen Temp", "°C")
            ]
            
            for i, (sensor_key, display_name, unit) in enumerate(sensors):
                if sensor_key in latest_data:
                    value = latest_data[sensor_key]
                    with cols[i % 4]:
                        st.metric(display_name, f"{value:.1f} {unit}")
                        
    def render_sensor_trends(self, data: pd.DataFrame):
        """Render sensor trend charts"""
        st.subheader("📈 Sensor Trends")
        
        if len(data) == 0:
            st.warning("No sensor data available")
            return
            
        # Create time series plots
        sensor_columns = [
            "steam_temperature", "steam_pressure", "turbine_vibration",
            "generator_temperature", "cooling_water_temp", "fuel_flow_rate",
            "oxygen_level", "load_factor"
        ]
        
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        if not available_sensors:
            st.warning("No sensor data columns found")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=len(available_sensors), cols=1,
            subplot_titles=available_sensors,
            vertical_spacing=0.05
        )
        
        for i, sensor in enumerate(available_sensors):
            fig.add_trace(
                go.Scatter(
                    x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                    y=data[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
            
        fig.update_layout(
            height=200 * len(available_sensors),
            showlegend=False,
            title="Sensor Readings Over Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_anomaly_detection(self, data: pd.DataFrame):
        """Render anomaly detection section"""
        st.subheader("🚨 Anomaly Detection")
        
        if len(data) == 0:
            st.warning("No data available for anomaly detection")
            return
            
        # Run anomaly detection if models are available
        if self.models:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display anomaly results
                self.display_anomaly_results(data)
                
            with col2:
                # Model performance metrics
                self.display_model_metrics()
                
        else:
            st.warning("No trained models available. Please train models first.")
            
    def display_anomaly_results(self, data: pd.DataFrame):
        """Display anomaly detection results"""
        try:
            # Use the most recent data for prediction
            recent_data = data.tail(100)  # Use last 100 records
            
            # Get predictions from each model
            model_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        predictions = model.predict(recent_data)
                        model_predictions[model_name] = predictions
                except Exception as e:
                    st.error(f"Error with {model_name}: {e}")
                    
            # Display ensemble prediction if available
            if "ensemble" in model_predictions:
                ensemble_preds = model_predictions["ensemble"]
                anomaly_count = np.sum(ensemble_preds)
                anomaly_rate = np.mean(ensemble_preds)
                
                # Anomaly summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies Detected", anomaly_count)
                with col2:
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
                with col3:
                    st.metric("Normal Rate", f"{1-anomaly_rate:.1%}")
                    
                # Anomaly timeline
                if len(ensemble_preds) > 0:
                    anomaly_df = pd.DataFrame({
                        'timestamp': recent_data.index if 'timestamp' not in recent_data.columns else recent_data['timestamp'],
                        'anomaly': ensemble_preds
                    })
                    
                    fig = px.line(
                        anomaly_df, 
                        x='timestamp', 
                        y='anomaly',
                        title="Anomaly Detection Timeline",
                        labels={'anomaly': 'Anomaly (1=Yes, 0=No)'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")
            
    def display_model_metrics(self):
        """Display model performance metrics"""
        st.subheader("Model Performance")
        
        # Placeholder for model metrics
        metrics = {
            "Isolation Forest": {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.88},
            "LSTM Autoencoder": {"Accuracy": 0.93, "Precision": 0.90, "Recall": 0.85},
            "Ensemble": {"Accuracy": 0.96, "Precision": 0.94, "Recall": 0.91}
        }
        
        for model_name, model_metrics in metrics.items():
            with st.expander(f"{model_name} Metrics"):
                for metric_name, value in model_metrics.items():
                    st.metric(metric_name, f"{value:.2f}")
                    
    def render_mlflow_integration(self):
        """Render MLflow integration section"""
        st.subheader("🔬 MLflow Experiments")
        
        if self.mlflow_manager:
            try:
                # Get recent experiments
                runs = self.mlflow_manager.search_runs(max_results=10)
                
                if runs:
                    # Create DataFrame for display
                    runs_df = pd.DataFrame([
                        {
                            "Run ID": run["run_id"][:8],
                            "Name": run["run_name"],
                            "Status": run["status"],
                            "Start Time": datetime.fromtimestamp(run["start_time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        }
                        for run in runs
                    ])
                    
                    st.dataframe(runs_df, use_container_width=True)
                    
                    # Display metrics for latest run
                    if runs:
                        latest_run = runs[0]
                        if "metrics" in latest_run and latest_run["metrics"]:
                            st.subheader("Latest Run Metrics")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                for metric_name, value in list(latest_run["metrics"].items())[:5]:
                                    st.metric(metric_name, f"{value:.4f}")
                                    
                            with col2:
                                for metric_name, value in list(latest_run["metrics"].items())[5:10]:
                                    st.metric(metric_name, f"{value:.4f}")
                                    
                else:
                    st.info("No MLflow experiments found")
                    
            except Exception as e:
                st.error(f"Error accessing MLflow: {e}")
        else:
            st.warning("MLflow manager not initialized")
            
    def render_model_training(self):
        """Render model training section"""
        st.subheader("🤖 Model Training")
        
        # Training controls
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Isolation Forest", "LSTM Autoencoder", "Ensemble"]
            )
            
        with col2:
            if st.button("Train Model"):
                self.train_model(model_type)
                
        # Training progress
        if "training_progress" in st.session_state:
            st.progress(st.session_state["training_progress"])
            
    def train_model(self, model_type: str):
        """Train selected model"""
        try:
            st.info(f"Training {model_type} model...")
            
            # Simulate training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training {model_type}: {i+1}%")
                time.sleep(0.01)
                
            st.success(f"{model_type} model training completed!")
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Controls")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["Simulated Data", "MQTT Stream", "Historical Data"]
        )
        
        # Refresh controls
        col1, col2 = st.sidebar.columns(2)
        with col1:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        with col2:
            refresh_interval = st.selectbox("Interval", [1, 5, 10, 30], index=1)
            
        # Model selection
        st.sidebar.subheader("Model Selection")
        selected_models = st.sidebar.multiselect(
            "Active Models",
            ["Isolation Forest", "LSTM Autoencoder", "Ensemble"],
            default=["Ensemble"]
        )
        
        # Alert settings
        st.sidebar.subheader("Alert Settings")
        anomaly_threshold = st.sidebar.slider(
            "Anomaly Threshold", 0.0, 1.0, 0.5, 0.01
        )
        
        enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)
        
        return {
            "data_source": data_source,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "selected_models": selected_models,
            "anomaly_threshold": anomaly_threshold,
            "enable_alerts": enable_alerts
        }
        
    def load_sample_data(self) -> pd.DataFrame:
        """Load sample data for demonstration"""
        try:
            # Try to load existing data
            if os.path.exists("data/sensor_data.csv"):
                df = pd.read_csv("data/sensor_data.csv")
                return df.tail(1000)  # Return last 1000 records
            else:
                # Generate sample data
                return self.generate_sample_data()
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
            
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample sensor data"""
        # Create time series
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='1min'
        )
        
        # Generate sensor data
        np.random.seed(42)
        n_samples = len(timestamps)
        
        data = {
            'timestamp': timestamps,
            'steam_temperature': 500 + np.random.normal(0, 10, n_samples),
            'steam_pressure': 150 + np.random.normal(0, 5, n_samples),
            'turbine_vibration': 1.0 + np.random.normal(0, 0.2, n_samples),
            'generator_temperature': 70 + np.random.normal(0, 3, n_samples),
            'cooling_water_temp': 30 + np.random.normal(0, 2, n_samples),
            'fuel_flow_rate': 60 + np.random.normal(0, 5, n_samples),
            'oxygen_level': 3.0 + np.random.normal(0, 0.3, n_samples),
            'load_factor': 85 + np.random.normal(0, 5, n_samples)
        }
        
        return pd.DataFrame(data)
        
    def run_dashboard(self):
        """Run the main dashboard"""
        # Initialize components
        self.initialize_components()
        
        # Render header
        self.render_header()
        
        # Render sidebar
        controls = self.render_sidebar()
        
        # Load data
        data = self.load_sample_data()
        
        if len(data) > 0:
            # Main content tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Real-time Monitoring", 
                "🚨 Anomaly Detection", 
                "🔬 MLflow Experiments",
                "🤖 Model Training"
            ])
            
            with tab1:
                self.render_sensor_overview(data)
                st.markdown("---")
                self.render_sensor_trends(data)
                
            with tab2:
                self.render_anomaly_detection(data)
                
            with tab3:
                self.render_mlflow_integration()
                
            with tab4:
                self.render_model_training()
                
        else:
            st.error("No data available. Please check your data sources.")
            
        # Auto refresh
        if controls["auto_refresh"]:
            time.sleep(controls["refresh_interval"])
            st.rerun()


def main():
    """Main function to run the dashboard"""
    try:
        dashboard = ThermalPlantDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
