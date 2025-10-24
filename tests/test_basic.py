"""
Basic tests for the mlops-thermal-plant package
"""
import pytest


def test_package_import():
    """Test that the main package can be imported"""
    try:
        import mlops_thermal_plant
        assert mlops_thermal_plant is not None
    except ImportError as e:
        pytest.fail(f"Failed to import package: {e}")


def test_models_import():
    """Test that models module can be imported"""
    try:
        from mlops_thermal_plant.core import models
        assert models is not None
    except ImportError as e:
        pytest.fail(f"Failed to import models: {e}")


def test_iot_import():
    """Test that IoT module can be imported"""
    try:
        from mlops_thermal_plant import iot
        assert iot is not None
    except ImportError as e:
        pytest.fail(f"Failed to import iot: {e}")
