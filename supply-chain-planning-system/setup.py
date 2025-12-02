"""Supply Chain Planning System - Setup Configuration."""

from setuptools import setup, find_packages

setup(
    name="supply-chain-planning-system",
    version="1.0.0",
    description="Unified end-to-end supply chain planning system orchestrating forecasting, inventory, pricing, network, sensing, and replenishment",
    author="Godson Kurishinkal",
    author_email="godson.kurishinkal@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "schedule>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "full": [
            "celery>=5.3.0",
            "redis>=5.0.0",
            "ortools>=9.6.0",
            "pulp>=2.7.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
)
