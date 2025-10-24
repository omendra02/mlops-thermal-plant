# Examples

This directory contains example scripts and configurations for using the MLOps Thermal Plant monitoring system.

## Generate Sample Data

Run the data generator to create sample sensor data:

```bash
python generate_data.py
```

This will create synthetic thermal plant sensor data for testing and development purposes.

## Configuration Examples

See the `config/*.example` files in the root config directory for example configurations:

- `config/database_config.yaml.example` - Database connection templates
- `config/mqtt_config.yaml.example` - MQTT broker configuration templates

Copy these files and remove the `.example` extension, then update with your actual credentials.
