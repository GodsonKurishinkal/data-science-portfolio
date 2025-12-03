"""Delivery Planning System - Setup Configuration."""
from setuptools import setup, find_packages

setup(
    name="delivery-planning-system",
    version="1.0.0",
    author="Godson Kurishinkal",
    author_email="godson@example.com",
    description="Comprehensive delivery planning with 3D bin packing and route optimization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GodsonKurishinkal/delivery-planning-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "pyright>=1.1.300",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
        "optimization": [
            "ortools>=9.6.0",
            "pulp>=2.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "delivery-planner=scripts.run_optimization:main",
        ],
    },
)
