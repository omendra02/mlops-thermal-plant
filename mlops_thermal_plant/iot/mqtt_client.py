"""
MQTT Client for IoT sensor data ingestion
"""

import json
import time
import logging
from typing import Dict, Any, Callable, Optional
import paho.mqtt.client as mqtt
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class ThermalPlantMQTTClient:
    """
    MQTT client for thermal power plant sensor data ingestion
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 client_id: str = "thermal_plant_client", username: str = None, 
                 password: str = None):
        """
        Initialize MQTT client
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            client_id: Unique client identifier
            username: MQTT username (if authentication required)
            password: MQTT password (if authentication required)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        self.data_callbacks = []
        self.is_connected = False
        self.subscribed_topics = set()
        
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            # Re-subscribe to topics
            for topic in self.subscribed_topics:
                client.subscribe(topic)
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
            
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.is_connected = False
        logger.warning(f"Disconnected from MQTT broker. Return code: {rc}")
        
    def _on_message(self, client, userdata, msg):
        """Callback for received MQTT messages"""
        try:
            # Parse JSON message
            data = json.loads(msg.payload.decode())
            data['timestamp'] = datetime.now().isoformat()
            data['topic'] = msg.topic
            
            logger.debug(f"Received data from {msg.topic}: {data}")
            
            # Call registered callbacks
            for callback in self.data_callbacks:
                callback(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message from {msg.topic}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {msg.topic}: {e}")
            
    def connect(self) -> bool:
        """
        Connect to MQTT broker
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        self.is_connected = False
        logger.info("Disconnected from MQTT broker")
        
    def subscribe(self, topic: str, qos: int = 1):
        """
        Subscribe to MQTT topic
        
        Args:
            topic: MQTT topic to subscribe to
            qos: Quality of Service level (0, 1, or 2)
        """
        if self.is_connected:
            self.client.subscribe(topic, qos)
            self.subscribed_topics.add(topic)
            logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.warning("Cannot subscribe: not connected to broker")
            
    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 1, retain: bool = False):
        """
        Publish data to MQTT topic
        
        Args:
            topic: MQTT topic to publish to
            payload: Data to publish (will be JSON serialized)
            qos: Quality of Service level
            retain: Whether to retain the message
        """
        if self.is_connected:
            try:
                json_payload = json.dumps(payload)
                result = self.client.publish(topic, json_payload, qos, retain)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.debug(f"Published to {topic}: {payload}")
                else:
                    logger.error(f"Failed to publish to {topic}. Error: {result.rc}")
                    
            except Exception as e:
                logger.error(f"Error publishing to {topic}: {e}")
        else:
            logger.warning("Cannot publish: not connected to broker")
            
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback function for received sensor data
        
        Args:
            callback: Function to call when sensor data is received
        """
        self.data_callbacks.append(callback)
        logger.info("Registered data callback")
        
    def unregister_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unregister callback function
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            logger.info("Unregistered data callback")
            
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status
        
        Returns:
            Dict containing connection status information
        """
        return {
            "connected": self.is_connected,
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "client_id": self.client_id,
            "subscribed_topics": list(self.subscribed_topics),
            "callbacks_registered": len(self.data_callbacks)
        }


class ThermalPlantMQTTSubscriber:
    """
    High-level MQTT subscriber for thermal plant sensor data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize subscriber with configuration
        
        Args:
            config: Configuration dictionary with MQTT settings
        """
        self.config = config
        self.client = ThermalPlantMQTTClient(
            broker_host=config.get("broker_host", "localhost"),
            broker_port=config.get("broker_port", 1883),
            client_id=config.get("client_id", "thermal_plant_subscriber"),
            username=config.get("username"),
            password=config.get("password")
        )
        
        self.sensor_data_buffer = []
        self.max_buffer_size = config.get("max_buffer_size", 1000)
        
    def start(self, topics: list):
        """
        Start MQTT subscriber
        
        Args:
            topics: List of MQTT topics to subscribe to
        """
        if self.client.connect():
            for topic in topics:
                self.client.subscribe(topic)
                
            # Register data callback
            self.client.register_data_callback(self._handle_sensor_data)
            logger.info(f"Started MQTT subscriber for topics: {topics}")
            return True
        else:
            logger.error("Failed to start MQTT subscriber")
            return False
            
    def stop(self):
        """Stop MQTT subscriber"""
        self.client.disconnect()
        logger.info("Stopped MQTT subscriber")
        
    def _handle_sensor_data(self, data: Dict[str, Any]):
        """
        Handle incoming sensor data
        
        Args:
            data: Sensor data dictionary
        """
        self.sensor_data_buffer.append(data)
        
        # Maintain buffer size
        if len(self.sensor_data_buffer) > self.max_buffer_size:
            self.sensor_data_buffer = self.sensor_data_buffer[-self.max_buffer_size:]
            
        logger.debug(f"Buffered sensor data. Buffer size: {len(self.sensor_data_buffer)}")
        
    def get_latest_data(self, count: int = 10) -> list:
        """
        Get latest sensor data from buffer
        
        Args:
            count: Number of latest records to return
            
        Returns:
            List of latest sensor data records
        """
        return self.sensor_data_buffer[-count:] if self.sensor_data_buffer else []
        
    def get_all_data(self) -> list:
        """
        Get all buffered sensor data
        
        Returns:
            List of all sensor data records
        """
        return self.sensor_data_buffer.copy()
        
    def clear_buffer(self):
        """Clear the sensor data buffer"""
        self.sensor_data_buffer.clear()
        logger.info("Cleared sensor data buffer")
