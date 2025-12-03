"""Fleet management for delivery vehicles."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import date

from src.vehicles.vehicle import Vehicle, VehicleType, VehicleStatus


@dataclass
class Fleet:
    """
    Manages a fleet of delivery vehicles.
    
    Provides methods for vehicle tracking, availability management,
    and fleet-wide operations.
    """
    vehicles: List[Vehicle] = field(default_factory=list)
    name: str = "Default Fleet"
    
    @property
    def size(self) -> int:
        """Total number of vehicles in fleet."""
        return len(self.vehicles)
    
    @property
    def available_vehicles(self) -> List[Vehicle]:
        """Get list of available vehicles."""
        return [v for v in self.vehicles if v.is_available]
    
    @property
    def in_use_vehicles(self) -> List[Vehicle]:
        """Get list of vehicles currently in use."""
        return [v for v in self.vehicles if v.status == VehicleStatus.IN_USE]
    
    @property
    def maintenance_vehicles(self) -> List[Vehicle]:
        """Get list of vehicles in maintenance."""
        return [v for v in self.vehicles if v.status == VehicleStatus.MAINTENANCE]
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet."""
        self.vehicles.append(vehicle)
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """Remove a vehicle from the fleet."""
        for i, v in enumerate(self.vehicles):
            if v.id == vehicle_id:
                self.vehicles.pop(i)
                return True
        return False
    
    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get a vehicle by ID."""
        for v in self.vehicles:
            if v.id == vehicle_id:
                return v
        return None
    
    def get_vehicles_by_type(self, vehicle_type: VehicleType) -> List[Vehicle]:
        """Get all vehicles of a specific type."""
        return [v for v in self.vehicles if v.vehicle_type == vehicle_type]
    
    def get_available_by_type(self, vehicle_type: VehicleType) -> List[Vehicle]:
        """Get available vehicles of a specific type."""
        return [v for v in self.available_vehicles if v.vehicle_type == vehicle_type]
    
    def find_vehicle_for_load(
        self,
        weight: float,
        volume: Optional[float] = None,
    ) -> Optional[Vehicle]:
        """
        Find an available vehicle suitable for a load.
        
        Args:
            weight: Required weight capacity (kg)
            volume: Required volume capacity (cubic cm, optional)
            
        Returns:
            Suitable vehicle or None
        """
        for v in self.available_vehicles:
            if v.max_weight >= weight:
                if volume is None or v.cargo_volume >= volume:
                    return v
        return None
    
    def get_capacity_summary(self) -> Dict:
        """Get summary of fleet capacity."""
        total_volume = sum(v.cargo_volume for v in self.vehicles)
        total_weight = sum(v.max_weight for v in self.vehicles)
        available_volume = sum(v.cargo_volume for v in self.available_vehicles)
        available_weight = sum(v.max_weight for v in self.available_vehicles)
        
        return {
            "total_vehicles": self.size,
            "available_vehicles": len(self.available_vehicles),
            "in_use_vehicles": len(self.in_use_vehicles),
            "maintenance_vehicles": len(self.maintenance_vehicles),
            "total_volume_m3": total_volume / 1_000_000,
            "total_weight_capacity_kg": total_weight,
            "available_volume_m3": available_volume / 1_000_000,
            "available_weight_capacity_kg": available_weight,
            "utilization_rate": (self.size - len(self.available_vehicles)) / max(1, self.size),
        }
    
    def get_type_breakdown(self) -> Dict:
        """Get breakdown of vehicles by type."""
        breakdown = {}
        for vtype in VehicleType:
            vehicles = self.get_vehicles_by_type(vtype)
            available = self.get_available_by_type(vtype)
            breakdown[vtype.value] = {
                "total": len(vehicles),
                "available": len(available),
                "in_use": len([v for v in vehicles if v.status == VehicleStatus.IN_USE]),
            }
        return breakdown
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics for all vehicles."""
        for vehicle in self.vehicles:
            vehicle.reset_daily_stats()
    
    def get_maintenance_needed(self) -> List[Vehicle]:
        """Get list of vehicles needing maintenance."""
        return [v for v in self.vehicles if v.needs_maintenance]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "capacity_summary": self.get_capacity_summary(),
            "type_breakdown": self.get_type_breakdown(),
        }


class FleetManager:
    """
    High-level fleet management operations.
    
    Handles fleet optimization, cost analysis, and reporting.
    """
    
    def __init__(self, fleet: Fleet):
        """
        Initialize fleet manager.
        
        Args:
            fleet: Fleet to manage
        """
        self.fleet = fleet
        self._daily_costs: Dict[str, float] = {}  # date -> cost
    
    def select_optimal_vehicle(
        self,
        weight: float,
        volume: float,
        distance_km: float,
        prefer_cost: bool = True,
    ) -> Optional[Vehicle]:
        """
        Select the optimal vehicle for a delivery.
        
        Args:
            weight: Required weight capacity
            volume: Required volume capacity
            distance_km: Trip distance
            prefer_cost: If True, prefer lower cost; else prefer efficiency
            
        Returns:
            Optimal vehicle or None
        """
        candidates = []
        
        for v in self.fleet.available_vehicles:
            if v.max_weight >= weight and v.cargo_volume >= volume:
                # Check fuel range
                if v.fuel_range < distance_km * 1.2:  # 20% margin
                    continue
                
                cost = v.estimate_cost(distance_km, distance_km / 40)  # Assume 40 km/h avg
                candidates.append((v, cost))
        
        if not candidates:
            return None
        
        if prefer_cost:
            # Sort by cost
            candidates.sort(key=lambda x: x[1])
        else:
            # Sort by efficiency (fuel per km)
            candidates.sort(key=lambda x: -x[0].fuel_efficiency)
        
        return candidates[0][0]
    
    def allocate_vehicles_for_routes(
        self,
        routes: List[Dict],
    ) -> Dict[str, Optional[str]]:
        """
        Allocate vehicles to multiple routes.
        
        Args:
            routes: List of route dictionaries with weight, volume, distance
            
        Returns:
            Mapping of route_id to vehicle_id
        """
        allocations = {}
        used_vehicles = set()
        
        # Sort routes by required capacity (largest first)
        sorted_routes = sorted(
            routes,
            key=lambda r: (r.get("weight", 0), r.get("volume", 0)),
            reverse=True,
        )
        
        for route in sorted_routes:
            route_id = route.get("id", "")
            weight = route.get("weight", 0)
            volume = route.get("volume", 0)
            distance = route.get("distance_km", 0)
            
            best_vehicle = None
            best_cost = float('inf')
            
            for v in self.fleet.available_vehicles:
                if v.id in used_vehicles:
                    continue
                
                if v.max_weight >= weight and v.cargo_volume >= volume:
                    if v.fuel_range < distance * 1.2:
                        continue
                    
                    cost = v.estimate_cost(distance, distance / 40)
                    if cost < best_cost:
                        best_cost = cost
                        best_vehicle = v
            
            if best_vehicle:
                allocations[route_id] = best_vehicle.id
                used_vehicles.add(best_vehicle.id)
            else:
                allocations[route_id] = None
        
        return allocations
    
    def calculate_daily_cost(self, operation_date: Optional[date] = None) -> Dict:
        """
        Calculate total daily operating costs.
        
        Args:
            operation_date: Date to calculate for (default: today)
            
        Returns:
            Cost breakdown dictionary
        """
        if operation_date is None:
            operation_date = date.today()
        
        fuel_cost = 0.0
        time_cost = 0.0
        
        for v in self.fleet.vehicles:
            # Estimate fuel used today
            fuel_used = v.total_km_today / v.fuel_efficiency
            fuel_cost += fuel_used * 1.5  # $1.50 per liter
            
            # Estimate operating time (rough: km at 40 km/h)
            hours = v.total_km_today / 40
            time_cost += hours * v.hourly_cost
        
        total_cost = fuel_cost + time_cost
        
        return {
            "date": operation_date.isoformat(),
            "fuel_cost": fuel_cost,
            "time_cost": time_cost,
            "total_cost": total_cost,
            "total_km": sum(v.total_km_today for v in self.fleet.vehicles),
            "total_deliveries": sum(v.total_deliveries_today for v in self.fleet.vehicles),
            "cost_per_km": total_cost / max(1, sum(v.total_km_today for v in self.fleet.vehicles)),
        }
    
    def get_utilization_report(self) -> Dict:
        """Generate fleet utilization report."""
        summary = self.fleet.get_capacity_summary()
        breakdown = self.fleet.get_type_breakdown()
        
        # Calculate efficiency metrics
        total_km = sum(v.total_km_today for v in self.fleet.vehicles)
        total_deliveries = sum(v.total_deliveries_today for v in self.fleet.vehicles)
        
        return {
            "fleet_summary": summary,
            "type_breakdown": breakdown,
            "performance": {
                "total_km_today": total_km,
                "total_deliveries_today": total_deliveries,
                "avg_km_per_vehicle": total_km / max(1, len(self.fleet.in_use_vehicles)),
                "avg_deliveries_per_vehicle": total_deliveries / max(1, len(self.fleet.in_use_vehicles)),
            },
            "maintenance_needed": len(self.fleet.get_maintenance_needed()),
        }
