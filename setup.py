from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlops-thermal-plant",
    version="0.1.0",
    author="Omendra Tomar",
    author_email="omendra26tomar@gmail.com",
    description="MLOps pipeline for thermal power plant monitoring and anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omendra02/mlops-thermal-plant",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "thermal-plant-train=scripts.train_models:main",
            "thermal-plant-dashboard=scripts.start_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mlops_thermal_plant": ["config/*.yaml", "config/*.conf"],
    },
)
