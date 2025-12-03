"""Vehicles module for fleet management."""
from .vehicle import Vehicle, VehicleType, VehicleStatus
from .fleet import Fleet, FleetManager

__all__ = [
    "Vehicle",
    "VehicleType",
    "VehicleStatus",
    "Fleet",
    "FleetManager",
]
