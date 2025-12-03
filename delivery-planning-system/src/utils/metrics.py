"""Performance metrics for packing and routing."""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PackingMetrics:
    """Metrics for evaluating bin packing performance."""
    
    total_volume: float = 0.0
    used_volume: float = 0.0
    total_weight: float = 0.0
    used_weight: float = 0.0
    num_boxes: int = 0
    packed_boxes: int = 0
    unpacked_boxes: int = 0
    
    @property
    def volume_utilization(self) -> float:
        """Volume utilization percentage."""
        if self.total_volume == 0:
            return 0.0
        return self.used_volume / self.total_volume
    
    @property
    def weight_utilization(self) -> float:
        """Weight utilization percentage."""
        if self.total_weight == 0:
            return 0.0
        return self.used_weight / self.total_weight
    
    @property
    def packing_success_rate(self) -> float:
        """Percentage of boxes successfully packed."""
        if self.num_boxes == 0:
            return 0.0
        return self.packed_boxes / self.num_boxes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_volume": self.total_volume,
            "used_volume": self.used_volume,
            "volume_utilization": self.volume_utilization,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,
            "weight_utilization": self.weight_utilization,
            "num_boxes": self.num_boxes,
            "packed_boxes": self.packed_boxes,
            "unpacked_boxes": self.unpacked_boxes,
            "packing_success_rate": self.packing_success_rate,
        }


@dataclass
class RouteMetrics:
    """Metrics for evaluating routing performance."""
    
    total_distance: float = 0.0
    total_time: float = 0.0  # minutes
    num_routes: int = 0
    num_stops: int = 0
    num_vehicles: int = 0
    unrouted_locations: int = 0
    
    @property
    def avg_distance_per_route(self) -> float:
        """Average distance per route."""
        if self.num_routes == 0:
            return 0.0
        return self.total_distance / self.num_routes
    
    @property
    def avg_time_per_route(self) -> float:
        """Average time per route in minutes."""
        if self.num_routes == 0:
            return 0.0
        return self.total_time / self.num_routes
    
    @property
    def avg_stops_per_route(self) -> float:
        """Average number of stops per route."""
        if self.num_routes == 0:
            return 0.0
        return self.num_stops / self.num_routes
    
    @property
    def routing_success_rate(self) -> float:
        """Percentage of locations successfully routed."""
        total = self.num_stops + self.unrouted_locations
        if total == 0:
            return 0.0
        return self.num_stops / total
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "num_routes": self.num_routes,
            "num_stops": self.num_stops,
            "num_vehicles": self.num_vehicles,
            "unrouted_locations": self.unrouted_locations,
            "avg_distance_per_route": self.avg_distance_per_route,
            "avg_time_per_route": self.avg_time_per_route,
            "avg_stops_per_route": self.avg_stops_per_route,
            "routing_success_rate": self.routing_success_rate,
        }


def calculate_packing_metrics(packing_result) -> PackingMetrics:
    """
    Calculate packing metrics from a packing result.
    
    Args:
        packing_result: PackingResult object
        
    Returns:
        PackingMetrics with calculated values
    """
    container = packing_result.container
    
    return PackingMetrics(
        total_volume=container.volume,
        used_volume=container.used_volume,
        total_weight=container.max_weight,
        used_weight=container.current_weight,
        num_boxes=packing_result.total_boxes,
        packed_boxes=packing_result.num_packed,
        unpacked_boxes=packing_result.num_unpacked,
    )


def calculate_route_metrics(vrp_solution) -> RouteMetrics:
    """
    Calculate routing metrics from a VRP solution.
    
    Args:
        vrp_solution: VRPSolution object
        
    Returns:
        RouteMetrics with calculated values
    """
    return RouteMetrics(
        total_distance=vrp_solution.total_distance,
        total_time=vrp_solution.total_time,
        num_routes=len([r for r in vrp_solution.routes if not r.is_empty()]),
        num_stops=vrp_solution.total_stops,
        num_vehicles=vrp_solution.num_vehicles_used,
        unrouted_locations=len(vrp_solution.unrouted),
    )
