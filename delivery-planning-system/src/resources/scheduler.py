"""Resource scheduling for delivery operations."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import date, time, datetime
from enum import Enum

from src.resources.driver import Driver, DriverPool, DriverStatus


class ShiftType(Enum):
    """Types of work shifts."""
    MORNING = "morning"      # 6:00 - 14:00
    DAY = "day"              # 8:00 - 18:00
    AFTERNOON = "afternoon"  # 14:00 - 22:00
    NIGHT = "night"          # 22:00 - 6:00
    FLEXIBLE = "flexible"


@dataclass
class Shift:
    """
    Represents a work shift.
    
    Attributes:
        id: Unique identifier
        shift_type: Type of shift
        start_time: Shift start time
        end_time: Shift end time
        date: Date of the shift
        required_drivers: Number of drivers needed
        assigned_drivers: List of assigned driver IDs
    """
    id: str = ""
    shift_type: ShiftType = ShiftType.DAY
    start_time: time = field(default_factory=lambda: time(8, 0))
    end_time: time = field(default_factory=lambda: time(18, 0))
    date: date = field(default_factory=date.today)
    required_drivers: int = 1
    assigned_drivers: List[str] = field(default_factory=list)
    vehicle_type_required: Optional[str] = None
    
    @property
    def duration_hours(self) -> float:
        """Calculate shift duration in hours."""
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        if end_minutes < start_minutes:  # Overnight shift
            end_minutes += 24 * 60
        return (end_minutes - start_minutes) / 60
    
    @property
    def is_fully_staffed(self) -> bool:
        """Check if all required positions are filled."""
        return len(self.assigned_drivers) >= self.required_drivers
    
    @property
    def vacancies(self) -> int:
        """Number of unfilled positions."""
        return max(0, self.required_drivers - len(self.assigned_drivers))
    
    def assign_driver(self, driver_id: str) -> bool:
        """Assign a driver to this shift."""
        if driver_id in self.assigned_drivers:
            return False
        if self.is_fully_staffed:
            return False
        self.assigned_drivers.append(driver_id)
        return True
    
    def remove_driver(self, driver_id: str) -> bool:
        """Remove a driver from this shift."""
        if driver_id in self.assigned_drivers:
            self.assigned_drivers.remove(driver_id)
            return True
        return False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "shift_type": self.shift_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "date": self.date.isoformat(),
            "duration_hours": self.duration_hours,
            "required_drivers": self.required_drivers,
            "assigned_drivers": self.assigned_drivers,
            "is_fully_staffed": self.is_fully_staffed,
            "vacancies": self.vacancies,
        }


@dataclass
class Assignment:
    """
    Represents a driver-to-route assignment.
    
    Attributes:
        id: Unique identifier
        driver_id: Assigned driver
        vehicle_id: Assigned vehicle
        route_id: Route to execute
        shift_id: Associated shift
        start_time: Assignment start time
        estimated_hours: Estimated duration
        status: Current status
    """
    id: str = ""
    driver_id: str = ""
    vehicle_id: str = ""
    route_id: str = ""
    shift_id: str = ""
    start_time: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    status: str = "pending"  # pending, active, completed, cancelled
    
    def start(self) -> None:
        """Start the assignment."""
        self.start_time = datetime.now()
        self.status = "active"
    
    def complete(self, hours: float) -> None:
        """Complete the assignment."""
        self.actual_hours = hours
        self.status = "completed"
    
    def cancel(self) -> None:
        """Cancel the assignment."""
        self.status = "cancelled"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "vehicle_id": self.vehicle_id,
            "route_id": self.route_id,
            "shift_id": self.shift_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "status": self.status,
        }


class ResourceScheduler:
    """
    Schedules drivers and resources for delivery operations.
    
    Handles:
    - Shift scheduling
    - Driver-to-route assignment
    - Workload balancing
    - Break management
    """
    
    def __init__(self, driver_pool: DriverPool):
        """
        Initialize the scheduler.
        
        Args:
            driver_pool: Pool of available drivers
        """
        self.driver_pool = driver_pool
        self.shifts: Dict[str, Shift] = {}
        self.assignments: Dict[str, Assignment] = {}
        self._assignment_counter = 0
    
    def create_shift(
        self,
        shift_type: ShiftType,
        shift_date: date,
        required_drivers: int = 1,
        vehicle_type: Optional[str] = None,
    ) -> Shift:
        """
        Create a new shift.
        
        Args:
            shift_type: Type of shift
            shift_date: Date for the shift
            required_drivers: Number of drivers needed
            vehicle_type: Required vehicle type (optional)
            
        Returns:
            Created Shift object
        """
        # Set times based on shift type
        shift_times = {
            ShiftType.MORNING: (time(6, 0), time(14, 0)),
            ShiftType.DAY: (time(8, 0), time(18, 0)),
            ShiftType.AFTERNOON: (time(14, 0), time(22, 0)),
            ShiftType.NIGHT: (time(22, 0), time(6, 0)),
            ShiftType.FLEXIBLE: (time(8, 0), time(18, 0)),
        }
        
        start, end = shift_times.get(shift_type, (time(8, 0), time(18, 0)))
        shift_id = f"SHIFT-{shift_date.isoformat()}-{shift_type.value}"
        
        shift = Shift(
            id=shift_id,
            shift_type=shift_type,
            start_time=start,
            end_time=end,
            date=shift_date,
            required_drivers=required_drivers,
            vehicle_type_required=vehicle_type,
        )
        
        self.shifts[shift_id] = shift
        return shift
    
    def auto_assign_shift(self, shift: Shift) -> List[str]:
        """
        Automatically assign drivers to a shift.
        
        Args:
            shift: Shift to staff
            
        Returns:
            List of assigned driver IDs
        """
        assigned = []
        
        # Find available drivers
        candidates = self.driver_pool.available_drivers
        
        # Filter by vehicle capability if specified
        if shift.vehicle_type_required:
            candidates = [
                d for d in candidates
                if d.can_drive_vehicle(shift.vehicle_type_required)
            ]
        
        # Filter by hours capacity
        candidates = [
            d for d in candidates
            if d.remaining_hours >= shift.duration_hours
        ]
        
        # Sort by hours worked (balance workload)
        candidates.sort(key=lambda d: d.hours_worked_today)
        
        # Assign drivers
        for driver in candidates:
            if shift.is_fully_staffed:
                break
            
            if shift.assign_driver(driver.id):
                assigned.append(driver.id)
        
        return assigned
    
    def create_assignment(
        self,
        driver_id: str,
        vehicle_id: str,
        route_id: str,
        estimated_hours: float,
        shift_id: Optional[str] = None,
    ) -> Optional[Assignment]:
        """
        Create a driver-to-route assignment.
        
        Args:
            driver_id: ID of the driver
            vehicle_id: ID of the vehicle
            route_id: ID of the route
            estimated_hours: Estimated route duration
            shift_id: Associated shift ID (optional)
            
        Returns:
            Created Assignment or None if driver unavailable
        """
        driver = self.driver_pool.get_driver(driver_id)
        if not driver or not driver.is_available:
            return None
        
        if driver.remaining_hours < estimated_hours:
            return None
        
        self._assignment_counter += 1
        assignment_id = f"ASSIGN-{self._assignment_counter:06d}"
        
        assignment = Assignment(
            id=assignment_id,
            driver_id=driver_id,
            vehicle_id=vehicle_id,
            route_id=route_id,
            shift_id=shift_id or "",
            estimated_hours=estimated_hours,
        )
        
        # Update driver state
        driver.assign_route(route_id, vehicle_id)
        
        self.assignments[assignment_id] = assignment
        return assignment
    
    def complete_assignment(
        self,
        assignment_id: str,
        actual_hours: float,
        deliveries: int = 0,
    ) -> bool:
        """
        Mark an assignment as complete.
        
        Args:
            assignment_id: ID of the assignment
            actual_hours: Actual hours worked
            deliveries: Number of deliveries made
            
        Returns:
            True if successful
        """
        assignment = self.assignments.get(assignment_id)
        if not assignment or assignment.status != "active":
            return False
        
        assignment.complete(actual_hours)
        
        # Update driver
        driver = self.driver_pool.get_driver(assignment.driver_id)
        if driver:
            driver.complete_route(actual_hours, deliveries)
        
        return True
    
    def get_driver_schedule(
        self,
        driver_id: str,
        schedule_date: Optional[date] = None,
    ) -> Dict:
        """
        Get schedule for a specific driver.
        
        Args:
            driver_id: Driver ID
            schedule_date: Date to get schedule for (default: today)
            
        Returns:
            Dictionary with driver's schedule information
        """
        if schedule_date is None:
            schedule_date = date.today()
        
        driver = self.driver_pool.get_driver(driver_id)
        if not driver:
            return {"error": "Driver not found"}
        
        # Find shifts
        driver_shifts = [
            s for s in self.shifts.values()
            if driver_id in s.assigned_drivers and s.date == schedule_date
        ]
        
        # Find assignments
        driver_assignments = [
            a for a in self.assignments.values()
            if a.driver_id == driver_id
        ]
        
        return {
            "driver": driver.to_dict(),
            "date": schedule_date.isoformat(),
            "shifts": [s.to_dict() for s in driver_shifts],
            "assignments": [a.to_dict() for a in driver_assignments],
        }
    
    def get_schedule_summary(
        self,
        schedule_date: Optional[date] = None,
    ) -> Dict:
        """
        Get summary of all schedules for a date.
        
        Args:
            schedule_date: Date to summarize (default: today)
            
        Returns:
            Summary dictionary
        """
        if schedule_date is None:
            schedule_date = date.today()
        
        day_shifts = [
            s for s in self.shifts.values()
            if s.date == schedule_date
        ]
        
        active_assignments = [
            a for a in self.assignments.values()
            if a.status == "active"
        ]
        
        return {
            "date": schedule_date.isoformat(),
            "total_shifts": len(day_shifts),
            "staffed_shifts": sum(1 for s in day_shifts if s.is_fully_staffed),
            "total_vacancies": sum(s.vacancies for s in day_shifts),
            "active_assignments": len(active_assignments),
            "driver_summary": self.driver_pool.get_workload_summary(),
        }
    
    def optimize_assignments(
        self,
        routes: List[Dict],
        vehicles: List[Dict],
    ) -> List[Assignment]:
        """
        Optimally assign drivers to routes and vehicles.
        
        Args:
            routes: List of routes with estimated hours
            vehicles: List of available vehicles with types
            
        Returns:
            List of created assignments
        """
        assignments = []
        
        # Sort routes by estimated hours (longer routes first)
        sorted_routes = sorted(routes, key=lambda r: r.get("estimated_hours", 0), reverse=True)
        
        for route in sorted_routes:
            route_hours = route.get("estimated_hours", 0)
            vehicle_type = route.get("vehicle_type", "medium_truck")
            
            # Find available vehicle
            available_vehicle = None
            for v in vehicles:
                if v.get("type") == vehicle_type and not v.get("assigned"):
                    available_vehicle = v
                    break
            
            if not available_vehicle:
                continue
            
            # Find best driver
            driver = self.driver_pool.find_best_driver(
                vehicle_type=vehicle_type,
                route_hours=route_hours,
                prefer_balanced=True,
            )
            
            if not driver:
                continue
            
            # Create assignment
            assignment = self.create_assignment(
                driver_id=driver.id,
                vehicle_id=available_vehicle.get("id", ""),
                route_id=route.get("id", ""),
                estimated_hours=route_hours,
            )
            
            if assignment:
                available_vehicle["assigned"] = True
                assignments.append(assignment)
        
        return assignments
