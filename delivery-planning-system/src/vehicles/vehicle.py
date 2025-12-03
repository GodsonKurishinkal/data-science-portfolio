"""Vehicle representation and management."""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid


class VehicleType(Enum):
    """Types of delivery vehicles."""
    SMALL_VAN = "small_van"
    MEDIUM_TRUCK = "medium_truck"
    LARGE_TRUCK = "large_truck"
    SEMI_TRAILER = "semi_trailer"


class VehicleStatus(Enum):
    """Current status of a vehicle."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"


# Vehicle specifications by type
VEHICLE_SPECS = {
    VehicleType.SMALL_VAN: {
        "length": 300, "width": 170, "height": 180,
        "max_weight": 1000, "fuel_efficiency": 10.0,
        "hourly_cost": 25.0,
    },
    VehicleType.MEDIUM_TRUCK: {
        "length": 450, "width": 220, "height": 230,
        "max_weight": 5000, "fuel_efficiency": 7.0,
        "hourly_cost": 40.0,
    },
    VehicleType.LARGE_TRUCK: {
        "length": 600, "width": 250, "height": 270,
        "max_weight": 10000, "fuel_efficiency": 5.0,
        "hourly_cost": 60.0,
    },
    VehicleType.SEMI_TRAILER: {
        "length": 1360, "width": 250, "height": 270,
        "max_weight": 25000, "fuel_efficiency": 3.5,
        "hourly_cost": 100.0,
    },
}


@dataclass
class Vehicle:
    """
    Represents a delivery vehicle.
    
    Attributes:
        id: Unique identifier
        vehicle_type: Type of vehicle
        license_plate: Vehicle license plate
        status: Current status
        cargo_length/width/height: Internal cargo dimensions (cm)
        max_weight: Maximum cargo weight (kg)
        current_driver_id: Currently assigned driver
        current_route_id: Currently assigned route
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vehicle_type: VehicleType = VehicleType.MEDIUM_TRUCK
    license_plate: str = ""
    status: VehicleStatus = VehicleStatus.AVAILABLE
    
    # Cargo dimensions (cm)
    cargo_length: float = 450.0
    cargo_width: float = 220.0
    cargo_height: float = 230.0
    max_weight: float = 5000.0
    
    # Operational parameters
    fuel_efficiency: float = 7.0  # km per liter
    hourly_cost: float = 40.0     # cost per hour
    current_fuel: float = 100.0   # liters
    tank_capacity: float = 100.0  # liters
    
    # Tracking
    current_driver_id: Optional[str] = None
    current_route_id: Optional[str] = None
    total_km_today: float = 0.0
    total_deliveries_today: int = 0
    
    # Maintenance
    next_maintenance_km: float = 10000.0
    total_km: float = 0.0
    
    @classmethod
    def from_type(cls, vehicle_type: VehicleType, vehicle_id: Optional[str] = None) -> 'Vehicle':
        """Create vehicle with specs from type."""
        specs = VEHICLE_SPECS.get(vehicle_type, VEHICLE_SPECS[VehicleType.MEDIUM_TRUCK])
        return cls(
            id=vehicle_id or str(uuid.uuid4())[:8],
            vehicle_type=vehicle_type,
            cargo_length=specs["length"],
            cargo_width=specs["width"],
            cargo_height=specs["height"],
            max_weight=specs["max_weight"],
            fuel_efficiency=specs["fuel_efficiency"],
            hourly_cost=specs["hourly_cost"],
        )
    
    @property
    def cargo_volume(self) -> float:
        """Calculate cargo volume in cubic cm."""
        return self.cargo_length * self.cargo_width * self.cargo_height
    
    @property
    def cargo_volume_m3(self) -> float:
        """Calculate cargo volume in cubic meters."""
        return self.cargo_volume / 1_000_000
    
    @property
    def is_available(self) -> bool:
        """Check if vehicle is available for assignment."""
        return (self.status == VehicleStatus.AVAILABLE and
                self.current_route_id is None)
    
    @property
    def fuel_range(self) -> float:
        """Calculate remaining range based on current fuel."""
        return self.current_fuel * self.fuel_efficiency
    
    @property
    def needs_maintenance(self) -> bool:
        """Check if vehicle needs maintenance."""
        return self.total_km >= self.next_maintenance_km
    
    def assign_route(self, route_id: str, driver_id: str) -> bool:
        """Assign a route and driver to this vehicle."""
        if not self.is_available:
            return False
        
        self.current_route_id = route_id
        self.current_driver_id = driver_id
        self.status = VehicleStatus.IN_USE
        return True
    
    def complete_route(self, km_traveled: float, deliveries: int) -> None:
        """Complete current route and update statistics."""
        self.total_km += km_traveled
        self.total_km_today += km_traveled
        self.total_deliveries_today += deliveries
        
        # Update fuel
        fuel_used = km_traveled / self.fuel_efficiency
        self.current_fuel = max(0, self.current_fuel - fuel_used)
        
        # Reset assignments
        self.current_route_id = None
        self.current_driver_id = None
        self.status = VehicleStatus.AVAILABLE
    
    def refuel(self, liters: Optional[float] = None) -> float:
        """
        Refuel the vehicle.
        
        Args:
            liters: Liters to add (None for full tank)
            
        Returns:
            Liters added
        """
        if liters is None:
            liters = self.tank_capacity - self.current_fuel
        
        actual = min(liters, self.tank_capacity - self.current_fuel)
        self.current_fuel += actual
        return actual
    
    def send_to_maintenance(self) -> None:
        """Send vehicle for maintenance."""
        self.status = VehicleStatus.MAINTENANCE
        self.current_route_id = None
        self.current_driver_id = None
    
    def complete_maintenance(self, km_interval: float = 10000.0) -> None:
        """Complete maintenance and return to service."""
        self.status = VehicleStatus.AVAILABLE
        self.next_maintenance_km = self.total_km + km_interval
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.total_km_today = 0.0
        self.total_deliveries_today = 0
    
    def estimate_cost(self, distance_km: float, hours: float) -> float:
        """
        Estimate cost for a trip.
        
        Args:
            distance_km: Distance in kilometers
            hours: Duration in hours
            
        Returns:
            Estimated cost
        """
        fuel_cost = (distance_km / self.fuel_efficiency) * 1.5  # Assume $1.50/liter
        time_cost = hours * self.hourly_cost
        return fuel_cost + time_cost
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "vehicle_type": self.vehicle_type.value,
            "license_plate": self.license_plate,
            "status": self.status.value,
            "cargo_dimensions": {
                "length": self.cargo_length,
                "width": self.cargo_width,
                "height": self.cargo_height,
            },
            "cargo_volume_m3": self.cargo_volume_m3,
            "max_weight": self.max_weight,
            "is_available": self.is_available,
            "current_fuel": self.current_fuel,
            "fuel_range": self.fuel_range,
            "current_driver_id": self.current_driver_id,
            "current_route_id": self.current_route_id,
            "total_km_today": self.total_km_today,
            "total_deliveries_today": self.total_deliveries_today,
            "needs_maintenance": self.needs_maintenance,
        }
    
    def __repr__(self) -> str:
        return f"Vehicle(id='{self.id}', type={self.vehicle_type.value}, status={self.status.value})"
