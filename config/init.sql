-- Database initialization script for Thermal Plant MLOps

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS thermal_plant_db;

-- Use the database
\c thermal_plant_db;

-- Create sensor readings table
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    plant_name VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    unit VARCHAR(20),
    quality_flag INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create anomaly predictions table
CREATE TABLE IF NOT EXISTS anomaly_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    plant_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    prediction INTEGER NOT NULL,
    confidence DECIMAL(5,4),
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    training_date TIMESTAMP,
    performance_metrics JSONB,
    hyperparameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system alerts table
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    plant_name VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_plant_name ON sensor_readings(plant_name);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_type ON sensor_readings(sensor_type);

CREATE INDEX IF NOT EXISTS idx_anomaly_predictions_timestamp ON anomaly_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_predictions_plant_name ON anomaly_predictions(plant_name);
CREATE INDEX IF NOT EXISTS idx_anomaly_predictions_model_type ON anomaly_predictions(model_type);

CREATE INDEX IF NOT EXISTS idx_system_alerts_timestamp ON system_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_alerts_plant_name ON system_alerts(plant_name);
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts(severity);

-- Create views for common queries
CREATE OR REPLACE VIEW latest_sensor_readings AS
SELECT 
    plant_name,
    sensor_type,
    value,
    unit,
    timestamp,
    ROW_NUMBER() OVER (PARTITION BY plant_name, sensor_type ORDER BY timestamp DESC) as rn
FROM sensor_readings
WHERE timestamp >= NOW() - INTERVAL '1 hour';

CREATE OR REPLACE VIEW recent_anomalies AS
SELECT 
    plant_name,
    model_type,
    COUNT(*) as anomaly_count,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as latest_anomaly
FROM anomaly_predictions
WHERE prediction = 1 
    AND timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY plant_name, model_type;

-- Insert sample data
INSERT INTO sensor_readings (timestamp, plant_name, sensor_type, value, unit) VALUES
(NOW() - INTERVAL '1 hour', 'Thermal Plant Alpha', 'steam_temperature', 485.5, '°C'),
(NOW() - INTERVAL '1 hour', 'Thermal Plant Alpha', 'steam_pressure', 152.3, 'bar'),
(NOW() - INTERVAL '1 hour', 'Thermal Plant Alpha', 'turbine_vibration', 1.2, 'mm/s'),
(NOW() - INTERVAL '1 hour', 'Thermal Plant Alpha', 'generator_temperature', 72.1, '°C');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO thermal_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO thermal_user;
