"""Driver and manpower management."""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime, time
import uuid


class DriverSkill(Enum):
    """Driver skill/certification levels."""
    JUNIOR = "junior"
    STANDARD = "standard"
    SENIOR = "senior"
    EXPERT = "expert"


class DriverStatus(Enum):
    """Current status of a driver."""
    AVAILABLE = "available"
    ON_ROUTE = "on_route"
    ON_BREAK = "on_break"
    OFF_DUTY = "off_duty"
    SICK = "sick"
    VACATION = "vacation"


# Vehicle types allowed by skill level
SKILL_VEHICLE_PERMISSIONS = {
    DriverSkill.JUNIOR: ["small_van"],
    DriverSkill.STANDARD: ["small_van", "medium_truck"],
    DriverSkill.SENIOR: ["small_van", "medium_truck", "large_truck"],
    DriverSkill.EXPERT: ["small_van", "medium_truck", "large_truck", "semi_trailer"],
}


@dataclass
class WorkingHours:
    """Working hours for a driver."""
    start: time = field(default_factory=lambda: time(8, 0))
    end: time = field(default_factory=lambda: time(18, 0))
    break_start: Optional[time] = field(default_factory=lambda: time(12, 0))
    break_duration_minutes: int = 30
    
    @property
    def total_hours(self) -> float:
        """Calculate total working hours."""
        start_minutes = self.start.hour * 60 + self.start.minute
        end_minutes = self.end.hour * 60 + self.end.minute
        total = (end_minutes - start_minutes) / 60
        if self.break_duration_minutes:
            total -= self.break_duration_minutes / 60
        return total
    
    def is_available_at(self, check_time: time) -> bool:
        """Check if driver is available at a specific time."""
        if check_time < self.start or check_time > self.end:
            return False
        
        if self.break_start:
            break_end_minutes = (self.break_start.hour * 60 + 
                                self.break_start.minute + 
                                self.break_duration_minutes)
            break_end = time(break_end_minutes // 60, break_end_minutes % 60)
            if self.break_start <= check_time < break_end:
                return False
        
        return True


@dataclass
class Driver:
    """
    Represents a delivery driver.
    
    Attributes:
        id: Unique identifier
        name: Driver name
        skill: Skill/certification level
        status: Current status
        working_hours: Regular working hours
        home_location: Driver's starting location
        current_vehicle_id: Currently assigned vehicle
        experience_years: Years of driving experience
        max_driving_hours: Maximum hours driver can drive per day
        overtime_allowed: Whether overtime is permitted
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    skill: DriverSkill = DriverSkill.STANDARD
    status: DriverStatus = DriverStatus.AVAILABLE
    working_hours: WorkingHours = field(default_factory=WorkingHours)
    home_location: Optional[str] = None
    current_vehicle_id: Optional[str] = None
    experience_years: int = 0
    max_driving_hours: float = 8.0
    overtime_allowed: bool = True
    max_overtime_hours: float = 2.0
    
    # Today's tracking
    hours_worked_today: float = 0.0
    deliveries_today: int = 0
    current_route_id: Optional[str] = None
    
    @property
    def allowed_vehicles(self) -> List[str]:
        """Get list of vehicle types this driver can operate."""
        return SKILL_VEHICLE_PERMISSIONS.get(self.skill, [])
    
    @property
    def remaining_hours(self) -> float:
        """Calculate remaining driving hours for today."""
        max_hours = self.max_driving_hours
        if self.overtime_allowed:
            max_hours += self.max_overtime_hours
        return max(0, max_hours - self.hours_worked_today)
    
    @property
    def is_available(self) -> bool:
        """Check if driver is available for assignment."""
        return (self.status == DriverStatus.AVAILABLE and
                self.remaining_hours > 0 and
                self.current_route_id is None)
    
    def can_drive_vehicle(self, vehicle_type: str) -> bool:
        """Check if driver is qualified to drive a vehicle type."""
        return vehicle_type.lower() in self.allowed_vehicles
    
    def assign_route(self, route_id: str, vehicle_id: str) -> bool:
        """Assign a route and vehicle to the driver."""
        if not self.is_available:
            return False
        
        self.current_route_id = route_id
        self.current_vehicle_id = vehicle_id
        self.status = DriverStatus.ON_ROUTE
        return True
    
    def complete_route(self, hours_driven: float, deliveries: int) -> None:
        """Mark route as complete and update statistics."""
        self.hours_worked_today += hours_driven
        self.deliveries_today += deliveries
        self.current_route_id = None
        self.status = DriverStatus.AVAILABLE
    
    def start_break(self) -> None:
        """Start break."""
        self.status = DriverStatus.ON_BREAK
    
    def end_break(self) -> None:
        """End break and return to available."""
        self.status = DriverStatus.AVAILABLE
    
    def end_shift(self) -> None:
        """End the driver's shift."""
        self.status = DriverStatus.OFF_DUTY
        self.current_route_id = None
        self.current_vehicle_id = None
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics for a new day."""
        self.hours_worked_today = 0.0
        self.deliveries_today = 0
        self.current_route_id = None
        if self.status not in [DriverStatus.SICK, DriverStatus.VACATION]:
            self.status = DriverStatus.AVAILABLE
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "skill": self.skill.value,
            "status": self.status.value,
            "allowed_vehicles": self.allowed_vehicles,
            "experience_years": self.experience_years,
            "hours_worked_today": self.hours_worked_today,
            "deliveries_today": self.deliveries_today,
            "remaining_hours": self.remaining_hours,
            "is_available": self.is_available,
            "current_vehicle_id": self.current_vehicle_id,
            "current_route_id": self.current_route_id,
        }
    
    def __repr__(self) -> str:
        return f"Driver(id='{self.id}', name='{self.name}', skill={self.skill.value})"


@dataclass
class DriverPool:
    """
    Manages a pool of drivers.
    
    Provides methods for finding available drivers, skill matching,
    and workload balancing.
    """
    drivers: List[Driver] = field(default_factory=list)
    
    @property
    def available_drivers(self) -> List[Driver]:
        """Get list of currently available drivers."""
        return [d for d in self.drivers if d.is_available]
    
    @property
    def total_remaining_capacity(self) -> float:
        """Total remaining driving hours across all drivers."""
        return sum(d.remaining_hours for d in self.available_drivers)
    
    def add_driver(self, driver: Driver) -> None:
        """Add a driver to the pool."""
        self.drivers.append(driver)
    
    def remove_driver(self, driver_id: str) -> bool:
        """Remove a driver from the pool."""
        for i, d in enumerate(self.drivers):
            if d.id == driver_id:
                self.drivers.pop(i)
                return True
        return False
    
    def get_driver(self, driver_id: str) -> Optional[Driver]:
        """Get a driver by ID."""
        for d in self.drivers:
            if d.id == driver_id:
                return d
        return None
    
    def find_drivers_for_vehicle(self, vehicle_type: str) -> List[Driver]:
        """Find available drivers qualified for a vehicle type."""
        return [
            d for d in self.available_drivers
            if d.can_drive_vehicle(vehicle_type)
        ]
    
    def find_best_driver(
        self,
        vehicle_type: str,
        route_hours: float,
        prefer_balanced: bool = True,
    ) -> Optional[Driver]:
        """
        Find the best driver for a route.
        
        Args:
            vehicle_type: Type of vehicle required
            route_hours: Estimated hours for the route
            prefer_balanced: If True, prefer drivers with fewer hours worked
            
        Returns:
            Best matching driver or None
        """
        candidates = [
            d for d in self.find_drivers_for_vehicle(vehicle_type)
            if d.remaining_hours >= route_hours
        ]
        
        if not candidates:
            return None
        
        if prefer_balanced:
            # Sort by hours worked (least first for balance)
            candidates.sort(key=lambda d: d.hours_worked_today)
        else:
            # Sort by skill level (higher first)
            skill_order = {DriverSkill.EXPERT: 0, DriverSkill.SENIOR: 1,
                          DriverSkill.STANDARD: 2, DriverSkill.JUNIOR: 3}
            candidates.sort(key=lambda d: skill_order.get(d.skill, 4))
        
        return candidates[0]
    
    def get_workload_summary(self) -> dict:
        """Get summary of workload distribution."""
        if not self.drivers:
            return {"total": 0, "available": 0, "on_route": 0}
        
        total_hours = sum(d.hours_worked_today for d in self.drivers)
        avg_hours = total_hours / len(self.drivers) if self.drivers else 0
        
        return {
            "total_drivers": len(self.drivers),
            "available": len(self.available_drivers),
            "on_route": len([d for d in self.drivers if d.status == DriverStatus.ON_ROUTE]),
            "on_break": len([d for d in self.drivers if d.status == DriverStatus.ON_BREAK]),
            "off_duty": len([d for d in self.drivers if d.status == DriverStatus.OFF_DUTY]),
            "total_hours_worked": total_hours,
            "average_hours": avg_hours,
            "total_deliveries": sum(d.deliveries_today for d in self.drivers),
        }
    
    def reset_all_daily_stats(self) -> None:
        """Reset daily stats for all drivers."""
        for driver in self.drivers:
            driver.reset_daily_stats()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "drivers": [d.to_dict() for d in self.drivers],
            "summary": self.get_workload_summary(),
        }
