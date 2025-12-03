"""Vehicles module for fleet management."""
from src.vehicles.vehicle import Vehicle, VehicleType, VehicleStatus
from src.vehicles.fleet import Fleet, FleetManager

__all__ = [
    "Vehicle",
    "VehicleType",
    "VehicleStatus",
    "Fleet",
    "FleetManager",
]
