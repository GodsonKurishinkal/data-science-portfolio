"""
Delivery Planning Orchestrator.

Main coordination module that brings together:
- 3D bin packing for vehicle loading
- Route optimization for delivery sequencing
- Resource management for driver assignment
- Vehicle allocation for fleet utilization
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

from ..packing import Box, Container, BinPacker, PackingResult
from ..routing import VRPSolver, DistanceMatrix, Location
from ..resources import DriverPool, ResourceScheduler
from ..vehicles import Fleet, FleetManager


@dataclass
class DeliveryOrder:
    """Represents a customer delivery order."""
    id: str
    customer_id: str
    destination: Location
    packages: List[Box]
    priority: int = 1
    time_window_start: Optional[float] = None  # Minutes from day start
    time_window_end: Optional[float] = None
    
    @property
    def total_weight(self) -> float:
        """Total weight of all packages."""
        return sum(p.weight for p in self.packages)
    
    @property
    def total_volume(self) -> float:
        """Total volume of all packages."""
        return sum(p.volume for p in self.packages)
    
    @property
    def num_packages(self) -> int:
        """Number of packages in order."""
        return len(self.packages)


@dataclass
class DeliveryPlan:
    """
    A complete delivery plan for a vehicle.
    
    Includes route, packing arrangement, and driver assignment.
    """
    id: str = ""
    vehicle_id: str = ""
    driver_id: str = ""
    route: List[int] = field(default_factory=list)  # Location indices
    orders: List[str] = field(default_factory=list)  # Order IDs
    packing_result: Optional[PackingResult] = None
    total_distance: float = 0.0
    total_time: float = 0.0  # Minutes
    estimated_cost: float = 0.0
    
    @property
    def num_stops(self) -> int:
        """Number of customer stops."""
        return max(0, len(self.route) - 2)  # Exclude depot start/end
    
    @property
    def utilization(self) -> float:
        """Vehicle space utilization."""
        if self.packing_result:
            return self.packing_result.utilization
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "driver_id": self.driver_id,
            "route": self.route,
            "orders": self.orders,
            "num_stops": self.num_stops,
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "estimated_cost": self.estimated_cost,
            "utilization": self.utilization,
            "packing": self.packing_result.to_dict() if self.packing_result else None,
        }


@dataclass
class PlanningResult:
    """
    Complete result of the delivery planning process.
    
    Contains all delivery plans, unassigned orders, and metrics.
    """
    plans: List[DeliveryPlan] = field(default_factory=list)
    unassigned_orders: List[str] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    total_cost: float = 0.0
    vehicles_used: int = 0
    computation_time: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if all orders are assigned."""
        return len(self.unassigned_orders) == 0
    
    @property
    def total_deliveries(self) -> int:
        """Total number of delivery stops."""
        return sum(p.num_stops for p in self.plans)
    
    @property
    def avg_utilization(self) -> float:
        """Average vehicle utilization."""
        if not self.plans:
            return 0.0
        return sum(p.utilization for p in self.plans) / len(self.plans)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "plans": [p.to_dict() for p in self.plans],
            "unassigned_orders": self.unassigned_orders,
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "total_cost": self.total_cost,
            "vehicles_used": self.vehicles_used,
            "total_deliveries": self.total_deliveries,
            "avg_utilization": self.avg_utilization,
            "is_complete": self.is_complete,
            "computation_time": self.computation_time,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DeliveryPlanner:
    """
    Main orchestrator for delivery planning.
    
    Coordinates:
    1. Order clustering and assignment to vehicles
    2. Route optimization for each vehicle
    3. 3D bin packing for vehicle loading
    4. Driver and resource assignment
    
    The planning process follows these steps:
    1. Analyze orders and estimate requirements
    2. Select and allocate vehicles
    3. Solve VRP to determine delivery routes
    4. Pack items in LIFO order (for each route)
    5. Assign drivers to routes
    6. Generate final delivery plans
    """
    
    def __init__(
        self,
        fleet: Fleet,
        driver_pool: DriverPool,
        depot_location: Location,
        packer: Optional[BinPacker] = None,
    ):
        """
        Initialize the delivery planner.
        
        Args:
            fleet: Available vehicle fleet
            driver_pool: Available drivers
            depot_location: Starting/ending depot location
            packer: Bin packer instance (created if not provided)
        """
        self.fleet = fleet
        self.fleet_manager = FleetManager(fleet)
        self.driver_pool = driver_pool
        self.scheduler = ResourceScheduler(driver_pool)
        self.depot = depot_location
        self.packer = packer or BinPacker()
        
        self._plan_counter = 0
        self._fixed_assignments: Dict[str, str] = {}
    
    def plan_deliveries(
        self,
        orders: List[DeliveryOrder],
        optimize_routes: bool = True,
        optimize_packing: bool = True,
    ) -> PlanningResult:
        """
        Create a complete delivery plan for all orders.
        
        Args:
            orders: List of delivery orders
            optimize_routes: Whether to optimize routes
            optimize_packing: Whether to optimize packing
            
        Returns:
            PlanningResult with all delivery plans
        """
        start_time = datetime.now()
        
        # Step 1: Prepare locations and distance matrix
        locations = [self.depot]
        order_location_map = {}  # order_id -> location_index
        
        for order in orders:
            locations.append(order.destination)
            order_location_map[order.id] = len(locations) - 1
        
        distance_matrix = DistanceMatrix(locations=locations)
        
        # Step 2: Solve VRP to get initial routes
        vrp_solver = VRPSolver(
            distance_matrix=distance_matrix,
            depot_idx=0,
            num_vehicles=len(self.fleet.available_vehicles),
            vehicle_capacity=10000,  # Will refine per vehicle
        )
        
        vrp_solution = vrp_solver.solve(
            algorithm="nearest_neighbor",
            improve=optimize_routes,
        )
        
        # Step 3: Create delivery plans for each route
        plans = []
        unassigned_orders = []
        
        for route in vrp_solution.routes:
            if route.is_empty():
                continue
            
            # Get orders for this route
            route_orders = []
            for stop_idx in route.stops[1:-1]:  # Exclude depot
                for order_id, loc_idx in order_location_map.items():
                    if loc_idx == stop_idx:
                        route_orders.append(order_id)
                        break
            
            if not route_orders:
                continue
            
            # Find order objects
            route_order_objs = [o for o in orders if o.id in route_orders]
            
            # Calculate total weight and volume
            total_weight = sum(o.total_weight for o in route_order_objs)
            total_volume = sum(o.total_volume for o in route_order_objs)
            
            # Select appropriate vehicle
            vehicle = self.fleet_manager.select_optimal_vehicle(
                weight=total_weight,
                volume=total_volume,
                distance_km=route.total_distance,
            )
            
            if not vehicle:
                unassigned_orders.extend(route_orders)
                continue
            
            # Create container from vehicle
            container = Container(
                id=vehicle.id,
                length=vehicle.cargo_length,
                width=vehicle.cargo_width,
                height=vehicle.cargo_height,
                max_weight=vehicle.max_weight,
            )
            
            # Collect all boxes with delivery sequence
            all_boxes = []
            for i, order in enumerate(route_order_objs):
                sequence = len(route_order_objs) - i  # LIFO: last delivery = sequence 1
                for box in order.packages:
                    box_copy = box.copy()
                    box_copy.sequence = sequence
                    box_copy.destination = order.id
                    all_boxes.append(box_copy)
            
            # Pack boxes
            packing_result = None
            if optimize_packing and all_boxes:
                packing_result = self.packer.pack(
                    boxes=all_boxes,
                    container=container,
                    optimize_sequence=True,
                )
                
                if packing_result.unpacked_boxes:
                    # Some boxes couldn't fit - may need larger vehicle
                    pass  # For now, continue with partial packing
            
            # Find driver
            driver = self.driver_pool.find_best_driver(
                vehicle_type=vehicle.vehicle_type.value,
                route_hours=route.total_time / 60,
            )
            
            # Create delivery plan
            self._plan_counter += 1
            plan = DeliveryPlan(
                id=f"PLAN-{self._plan_counter:06d}",
                vehicle_id=vehicle.id,
                driver_id=driver.id if driver else "",
                route=route.stops,
                orders=route_orders,
                packing_result=packing_result,
                total_distance=route.total_distance,
                total_time=route.total_time,
                estimated_cost=vehicle.estimate_cost(
                    route.total_distance,
                    route.total_time / 60,
                ),
            )
            
            plans.append(plan)
            
            # Update vehicle and driver status
            if driver:
                vehicle.assign_route(plan.id, driver.id)
                driver.assign_route(plan.id, vehicle.id)
        
        # Add any unrouted locations from VRP
        for unrouted_idx in vrp_solution.unrouted:
            for order_id, loc_idx in order_location_map.items():
                if loc_idx == unrouted_idx:
                    unassigned_orders.append(order_id)
        
        # Calculate totals
        computation_time = (datetime.now() - start_time).total_seconds()
        
        result = PlanningResult(
            plans=plans,
            unassigned_orders=list(set(unassigned_orders)),
            total_distance=sum(p.total_distance for p in plans),
            total_time=sum(p.total_time for p in plans),
            total_cost=sum(p.estimated_cost for p in plans),
            vehicles_used=len(plans),
            computation_time=computation_time,
        )
        
        return result
    
    def replan_with_constraints(
        self,
        orders: List[DeliveryOrder],
        fixed_assignments: Dict[str, str] = None,
        excluded_vehicles: List[str] = None,
    ) -> PlanningResult:
        """
        Replan with specific constraints.
        
        Args:
            orders: Delivery orders
            fixed_assignments: Order ID -> Vehicle ID fixed assignments
            excluded_vehicles: Vehicle IDs to exclude
            
        Returns:
            Updated planning result
        """
        # Store fixed assignments for use during planning
        self._fixed_assignments = fixed_assignments or {}
        
        # Temporarily mark excluded vehicles as unavailable
        excluded = excluded_vehicles or []
        original_status = {}
        
        for v_id in excluded:
            vehicle = self.fleet.get_vehicle(v_id)
            if vehicle:
                original_status[v_id] = vehicle.status
                vehicle.status = VehicleStatus.OUT_OF_SERVICE
        
        try:
            result = self.plan_deliveries(orders)
        finally:
            # Restore original status
            for v_id, status in original_status.items():
                vehicle = self.fleet.get_vehicle(v_id)
                if vehicle:
                    vehicle.status = status
        
        return result
    
    def get_packing_visualization_data(self, plan: DeliveryPlan) -> Dict:
        """
        Get data for 3D visualization of packing.
        
        Args:
            plan: Delivery plan with packing result
            
        Returns:
            Visualization data dictionary
        """
        if not plan.packing_result:
            return {"error": "No packing result available"}
        
        result = plan.packing_result
        
        # Get container dimensions
        container = result.container
        
        boxes_data = []
        for box in result.packed_boxes:
            boxes_data.append({
                "id": box.id,
                "position": {
                    "x": box.position.x,
                    "y": box.position.y,
                    "z": box.position.z,
                },
                "dimensions": {
                    "length": box.dimensions.length,
                    "width": box.dimensions.width,
                    "height": box.dimensions.height,
                },
                "weight": box.weight,
                "sequence": box.sequence,
                "destination": box.destination,
                "rotated": box.rotated,
            })
        
        return {
            "container": {
                "id": container.id,
                "length": container.length,
                "width": container.width,
                "height": container.height,
            },
            "boxes": boxes_data,
            "packing_sequence": result.packing_sequence,
            "utilization": result.utilization,
            "total_boxes": result.num_packed,
        }


# Import for type hint only
from ..vehicles.vehicle import VehicleStatus
