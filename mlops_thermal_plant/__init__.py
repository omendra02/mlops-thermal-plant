"""
MLOps Thermal Plant Monitor

A comprehensive MLOps pipeline for thermal power plant monitoring and anomaly detection.
Includes IoT integration, real-time monitoring, and automated model management.
"""

__version__ = "1.0.0"
__author__ = "MLOps Thermal Plant Team"
__email__ = "team@thermal-mlops.com"

from .core import models, data, monitoring
from .iot import mqtt_client, sensor_simulator
from .dashboard import dashboard_app

__all__ = [
    "models",
    "data", 
    "monitoring",
    "mqtt_client",
    "sensor_simulator",
    "dashboard_app"
]
