# Multi-stage Docker build for Thermal Plant MLOps

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/experiments

# Expose ports
EXPOSE 8501 5000 8080

# Set default command for development
CMD ["streamlit", "run", "mlops_thermal_plant/dashboard/dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 3: Production image
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash thermal_user

# Install production dependencies only
COPY requirements.txt /app/
WORKDIR /app

# Install only production dependencies (exclude dev tools)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip uninstall -y pytest pytest-cov black flake8 mypy

# Copy application code
COPY --chown=thermal_user:thermal_user . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/logs /app/experiments && \
    chown -R thermal_user:thermal_user /app

# Switch to non-root user
USER thermal_user

# Expose ports
EXPOSE 8501 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set default command for production
CMD ["streamlit", "run", "mlops_thermal_plant/dashboard/dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
