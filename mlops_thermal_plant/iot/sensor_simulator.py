"""
Sensor Simulator for Thermal Power Plant IoT Data
"""

import time
import random
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from .mqtt_client import ThermalPlantMQTTClient

logger = logging.getLogger(__name__)


class ThermalPlantSensorSimulator:
    """
    Simulates sensor data for thermal power plant equipment
    """
    
    def __init__(self, plant_config: Dict[str, Any], mqtt_client: Optional[ThermalPlantMQTTClient] = None):
        """
        Initialize sensor simulator
        
        Args:
            plant_config: Configuration for the thermal plant
            mqtt_client: MQTT client for publishing data (optional)
        """
        self.plant_config = plant_config
        self.mqtt_client = mqtt_client
        self.is_running = False
        self.simulation_thread = None
        
        # Plant parameters
        self.plant_name = plant_config.get("plant_name", "Thermal Plant Alpha")
        self.capacity_mw = plant_config.get("capacity_mw", 500)
        self.fuel_type = plant_config.get("fuel_type", "Coal")
        
        # Sensor configurations
        self.sensors = self._initialize_sensors()
        
        # Simulation parameters
        self.update_interval = plant_config.get("update_interval", 1.0)  # seconds
        self.anomaly_probability = plant_config.get("anomaly_probability", 0.05)
        
    def _initialize_sensors(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize sensor configurations based on plant type and capacity
        
        Returns:
            Dictionary of sensor configurations
        """
        # Base sensor configurations (normal operating ranges)
        sensors = {
            "steam_temperature": {
                "unit": "°C",
                "normal_range": (480, 520),
                "critical_range": (450, 550),
                "noise_level": 2.0,
                "trend_cycle_hours": 24,
                "anomaly_multiplier": 1.5
            },
            "steam_pressure": {
                "unit": "bar",
                "normal_range": (140, 160),
                "critical_range": (120, 180),
                "noise_level": 1.0,
                "trend_cycle_hours": 12,
                "anomaly_multiplier": 1.3
            },
            "turbine_vibration": {
                "unit": "mm/s",
                "normal_range": (0.5, 2.0),
                "critical_range": (0.0, 5.0),
                "noise_level": 0.1,
                "trend_cycle_hours": 8,
                "anomaly_multiplier": 2.0
            },
            "generator_temperature": {
                "unit": "°C",
                "normal_range": (60, 80),
                "critical_range": (40, 100),
                "noise_level": 1.0,
                "trend_cycle_hours": 6,
                "anomaly_multiplier": 1.4
            },
            "cooling_water_temp": {
                "unit": "°C",
                "normal_range": (25, 35),
                "critical_range": (15, 45),
                "noise_level": 0.5,
                "trend_cycle_hours": 12,
                "anomaly_multiplier": 1.2
            },
            "fuel_flow_rate": {
                "unit": "kg/s",
                "normal_range": (50, 80),
                "critical_range": (30, 100),
                "noise_level": 2.0,
                "trend_cycle_hours": 4,
                "anomaly_multiplier": 1.3
            },
            "oxygen_level": {
                "unit": "%",
                "normal_range": (2.5, 3.5),
                "critical_range": (2.0, 4.0),
                "noise_level": 0.1,
                "trend_cycle_hours": 2,
                "anomaly_multiplier": 1.5
            },
            "load_factor": {
                "unit": "%",
                "normal_range": (70, 95),
                "critical_range": (50, 100),
                "noise_level": 1.0,
                "trend_cycle_hours": 8,
                "anomaly_multiplier": 1.2
            }
        }
        
        # Adjust sensor ranges based on plant capacity
        capacity_factor = self.capacity_mw / 500.0  # Normalize to 500MW reference
        
        for sensor_name, config in sensors.items():
            if "rate" in sensor_name or "flow" in sensor_name:
                # Flow rates scale with capacity
                config["normal_range"] = tuple(x * capacity_factor for x in config["normal_range"])
                config["critical_range"] = tuple(x * capacity_factor for x in config["critical_range"])
                
        return sensors
        
    def _generate_sensor_value(self, sensor_name: str, timestamp: datetime) -> float:
        """
        Generate a realistic sensor value
        
        Args:
            sensor_name: Name of the sensor
            timestamp: Current timestamp
            
        Returns:
            Generated sensor value
        """
        sensor_config = self.sensors[sensor_name]
        
        # Base value (center of normal range)
        normal_min, normal_max = sensor_config["normal_range"]
        base_value = (normal_min + normal_max) / 2
        
        # Add cyclic trend based on time of day/operating cycles
        cycle_hours = sensor_config["trend_cycle_hours"]
        cycle_phase = (timestamp.hour + timestamp.minute / 60.0) / cycle_hours * 2 * np.pi
        
        # Trend amplitude (small percentage of normal range)
        trend_amplitude = (normal_max - normal_min) * 0.1
        trend = np.sin(cycle_phase) * trend_amplitude
        
        # Add noise
        noise = np.random.normal(0, sensor_config["noise_level"])
        
        # Calculate final value
        value = base_value + trend + noise
        
        # Apply anomaly if probability is met
        if random.random() < self.anomaly_probability:
            anomaly_type = random.choice(["spike", "drift", "noise_burst"])
            value = self._apply_anomaly(value, sensor_config, anomaly_type)
            
        return round(value, 2)
        
    def _apply_anomaly(self, value: float, sensor_config: Dict[str, Any], anomaly_type: str) -> float:
        """
        Apply different types of anomalies to sensor values
        
        Args:
            value: Original sensor value
            sensor_config: Sensor configuration
            anomaly_type: Type of anomaly to apply
            
        Returns:
            Anomalous sensor value
        """
        normal_min, normal_max = sensor_config["normal_range"]
        critical_min, critical_max = sensor_config["critical_range"]
        multiplier = sensor_config["anomaly_multiplier"]
        
        if anomaly_type == "spike":
            # Sudden spike up or down
            direction = random.choice([-1, 1])
            spike_magnitude = (normal_max - normal_min) * 0.5 * multiplier
            return value + direction * spike_magnitude
            
        elif anomaly_type == "drift":
            # Gradual drift outside normal range
            drift_direction = random.choice([-1, 1])
            drift_magnitude = (normal_max - normal_min) * 0.3 * multiplier
            return value + drift_direction * drift_magnitude
            
        elif anomaly_type == "noise_burst":
            # High-frequency noise burst
            noise_burst = np.random.normal(0, sensor_config["noise_level"] * 3)
            return value + noise_burst
            
        return value
        
    def _generate_sensor_data(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Generate complete sensor data for all sensors
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing all sensor readings
        """
        sensor_data = {
            "timestamp": timestamp.isoformat(),
            "plant_name": self.plant_name,
            "fuel_type": self.fuel_type,
            "capacity_mw": self.capacity_mw,
        }
        
        # Generate values for each sensor
        for sensor_name in self.sensors.keys():
            sensor_data[sensor_name] = self._generate_sensor_value(sensor_name, timestamp)
            
        # Calculate derived metrics
        sensor_data.update(self._calculate_derived_metrics(sensor_data))
        
        return sensor_data
        
    def _calculate_derived_metrics(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate derived metrics from sensor data
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            Dictionary of derived metrics
        """
        derived = {}
        
        # Heat rate calculation (simplified)
        steam_temp = sensor_data["steam_temperature"]
        steam_pressure = sensor_data["steam_pressure"]
        load_factor = sensor_data["load_factor"]
        
        # Simplified heat rate formula (lower is better)
        base_heat_rate = 10000
        temp_factor = (520 - steam_temp) * 10  # Higher temp = better efficiency
        pressure_factor = (steam_pressure - 150) * 5  # Higher pressure = better efficiency
        load_factor_penalty = (100 - load_factor) * 20  # Lower load = worse efficiency
        
        heat_rate = base_heat_rate - temp_factor - pressure_factor + load_factor_penalty
        derived["heat_rate"] = round(heat_rate, 2)
        
        # Efficiency calculation
        efficiency = max(0, min(100, 45 - (heat_rate - 8500) / 100))
        derived["efficiency"] = round(efficiency, 2)
        
        # Equipment health score
        health_score = self._calculate_health_score(sensor_data)
        derived["equipment_health"] = round(health_score, 2)
        
        return derived
        
    def _calculate_health_score(self, sensor_data: Dict[str, Any]) -> float:
        """
        Calculate overall equipment health score (0-100)
        
        Args:
            sensor_data: Sensor data
            
        Returns:
            Health score (0-100, where 100 is perfect health)
        """
        health_factors = []
        
        for sensor_name, value in sensor_data.items():
            if sensor_name in self.sensors:
                sensor_config = self.sensors[sensor_name]
                normal_min, normal_max = sensor_config["normal_range"]
                critical_min, critical_max = sensor_config["critical_range"]
                
                # Calculate health factor for this sensor
                if normal_min <= value <= normal_max:
                    # Perfect health in normal range
                    factor = 100
                elif critical_min <= value <= critical_max:
                    # Degraded health in critical range
                    if value < normal_min:
                        factor = 100 * (value - critical_min) / (normal_min - critical_min)
                    else:
                        factor = 100 * (critical_max - value) / (critical_max - normal_max)
                else:
                    # Poor health outside critical range
                    factor = 0
                    
                health_factors.append(factor)
                
        return np.mean(health_factors) if health_factors else 50
        
    def _simulation_loop(self):
        """Main simulation loop"""
        logger.info("Starting sensor simulation loop")
        
        while self.is_running:
            try:
                timestamp = datetime.now()
                sensor_data = self._generate_sensor_data(timestamp)
                
                # Publish to MQTT if client is available
                if self.mqtt_client and self.mqtt_client.is_connected:
                    topic = f"thermal_plant/{self.plant_name.lower().replace(' ', '_')}/sensors"
                    self.mqtt_client.publish(topic, sensor_data)
                    
                # Log data periodically
                if timestamp.second % 30 == 0:  # Log every 30 seconds
                    logger.debug(f"Generated sensor data: {sensor_data}")
                    
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(self.update_interval)
                
        logger.info("Sensor simulation loop stopped")
        
    def start_simulation(self):
        """Start the sensor simulation"""
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
            logger.info(f"Started sensor simulation for {self.plant_name}")
        else:
            logger.warning("Simulation is already running")
            
    def stop_simulation(self):
        """Stop the sensor simulation"""
        if self.is_running:
            self.is_running = False
            if self.simulation_thread:
                self.simulation_thread.join(timeout=5)
            logger.info("Stopped sensor simulation")
        else:
            logger.warning("Simulation is not running")
            
    def get_current_data(self) -> Dict[str, Any]:
        """
        Get current simulated sensor data
        
        Returns:
            Current sensor data dictionary
        """
        timestamp = datetime.now()
        return self._generate_sensor_data(timestamp)
        
    def get_sensor_config(self, sensor_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific sensor
        
        Args:
            sensor_name: Name of the sensor
            
        Returns:
            Sensor configuration dictionary
        """
        return self.sensors.get(sensor_name, {})
        
    def get_all_sensor_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration for all sensors
        
        Returns:
            Dictionary of all sensor configurations
        """
        return self.sensors.copy()


class MultiPlantSimulator:
    """
    Simulator for multiple thermal power plants
    """
    
    def __init__(self, plants_config: List[Dict[str, Any]], mqtt_client: Optional[ThermalPlantMQTTClient] = None):
        """
        Initialize multi-plant simulator
        
        Args:
            plants_config: List of plant configurations
            mqtt_client: MQTT client for publishing data
        """
        self.plants_config = plants_config
        self.mqtt_client = mqtt_client
        self.plant_simulators = {}
        
        # Initialize simulators for each plant
        for plant_config in plants_config:
            plant_name = plant_config["plant_name"]
            self.plant_simulators[plant_name] = ThermalPlantSensorSimulator(
                plant_config, mqtt_client
            )
            
    def start_all_simulations(self):
        """Start simulation for all plants"""
        for plant_name, simulator in self.plant_simulators.items():
            simulator.start_simulation()
            logger.info(f"Started simulation for {plant_name}")
            
    def stop_all_simulations(self):
        """Stop simulation for all plants"""
        for plant_name, simulator in self.plant_simulators.items():
            simulator.stop_simulation()
            logger.info(f"Stopped simulation for {plant_name}")
            
    def get_plant_simulator(self, plant_name: str) -> Optional[ThermalPlantSensorSimulator]:
        """
        Get simulator for a specific plant
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            Plant simulator instance or None
        """
        return self.plant_simulators.get(plant_name)
        
    def get_all_current_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current data from all plants
        
        Returns:
            Dictionary mapping plant names to their current sensor data
        """
        return {
            plant_name: simulator.get_current_data()
            for plant_name, simulator in self.plant_simulators.items()
        }
