"""Vehicle Routing Problem (VRP) Solver."""
from dataclasses import dataclass, field
from typing import List
from enum import Enum

from .distance import DistanceMatrix


class VRPType(Enum):
    """Types of VRP problems."""
    BASIC = "basic"                    # Basic VRP
    CVRP = "cvrp"                      # Capacitated VRP
    VRPTW = "vrptw"                    # VRP with Time Windows
    VRPPD = "vrppd"                    # VRP with Pickup and Delivery
    MDVRP = "mdvrp"                    # Multi-Depot VRP


@dataclass
class Route:
    """
    Represents a delivery route for a vehicle.
    
    Attributes:
        vehicle_id: ID of the assigned vehicle
        stops: List of location indices in order
        total_distance: Total route distance
        total_time: Total route time (travel + service)
        total_demand: Total demand/load on this route
    """
    vehicle_id: str = ""
    stops: List[int] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    total_demand: float = 0.0
    arrival_times: List[float] = field(default_factory=list)
    
    @property
    def num_stops(self) -> int:
        """Number of stops (excluding depot)."""
        return max(0, len(self.stops) - 2)  # Exclude start and end depot
    
    def is_empty(self) -> bool:
        """Check if route has no customer stops."""
        return self.num_stops == 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vehicle_id": self.vehicle_id,
            "stops": self.stops,
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "total_demand": self.total_demand,
            "num_stops": self.num_stops,
            "arrival_times": self.arrival_times,
        }


@dataclass
class VRPSolution:
    """
    Solution to a Vehicle Routing Problem.
    
    Attributes:
        routes: List of routes for each vehicle
        total_distance: Sum of all route distances
        total_time: Sum of all route times
        unrouted: Locations that couldn't be routed
    """
    routes: List[Route] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    unrouted: List[int] = field(default_factory=list)
    iterations: int = 0
    computation_time: float = 0.0
    
    @property
    def num_vehicles_used(self) -> int:
        """Number of vehicles with non-empty routes."""
        return sum(1 for r in self.routes if not r.is_empty())
    
    @property
    def total_stops(self) -> int:
        """Total number of customer stops."""
        return sum(r.num_stops for r in self.routes)
    
    @property
    def is_complete(self) -> bool:
        """Check if all locations are routed."""
        return len(self.unrouted) == 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "routes": [r.to_dict() for r in self.routes],
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "num_vehicles_used": self.num_vehicles_used,
            "total_stops": self.total_stops,
            "unrouted": self.unrouted,
            "is_complete": self.is_complete,
            "iterations": self.iterations,
        }


class VRPSolver:
    """
    Solver for Vehicle Routing Problems.
    
    Implements multiple algorithms:
    - Nearest Neighbor (construction heuristic)
    - Savings Algorithm (Clarke-Wright)
    - 2-opt and Or-opt (improvement heuristics)
    """
    
    def __init__(
        self,
        distance_matrix: DistanceMatrix,
        depot_idx: int = 0,
        num_vehicles: int = 10,
        vehicle_capacity: float = float('inf'),
        max_route_time: float = 480.0,  # 8 hours in minutes
    ):
        """
        Initialize the VRP solver.
        
        Args:
            distance_matrix: Distance/time matrix for locations
            depot_idx: Index of the depot location
            num_vehicles: Number of available vehicles
            vehicle_capacity: Capacity of each vehicle
            max_route_time: Maximum route duration in minutes
        """
        self.dm = distance_matrix
        self.depot_idx = depot_idx
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.max_route_time = max_route_time
    
    def solve(
        self,
        algorithm: str = "nearest_neighbor",
        improve: bool = True,
        max_iterations: int = 1000,
    ) -> VRPSolution:
        """
        Solve the VRP using specified algorithm.
        
        Args:
            algorithm: Algorithm to use ('nearest_neighbor', 'savings')
            improve: Whether to apply improvement heuristics
            max_iterations: Maximum iterations for improvement
            
        Returns:
            VRPSolution with routes
        """
        # Build initial solution
        if algorithm == "savings":
            solution = self._savings_algorithm()
        else:
            solution = self._nearest_neighbor()
        
        # Apply improvement heuristics
        if improve:
            solution = self._improve_solution(solution, max_iterations)
        
        # Calculate final metrics
        self._calculate_solution_metrics(solution)
        
        return solution
    
    def _nearest_neighbor(self) -> VRPSolution:
        """
        Construct routes using Nearest Neighbor heuristic.
        
        For each vehicle, repeatedly add the nearest unvisited customer
        until capacity or time constraints are violated.
        """
        routes = []
        visited = {self.depot_idx}
        unrouted = []
        
        for v in range(self.num_vehicles):
            route = Route(vehicle_id=f"V{v+1:03d}")
            route.stops = [self.depot_idx]
            route_demand = 0.0
            route_time = 0.0
            
            current = self.depot_idx
            
            while True:
                # Find nearest unvisited customer
                best_next = None
                best_dist = float('inf')
                
                for i in range(self.dm.size):
                    if i in visited:
                        continue
                    
                    dist = self.dm.get_distance(current, i)
                    loc = self.dm.locations[i]
                    
                    # Check constraints
                    new_demand = route_demand + loc.demand
                    new_time = route_time + self.dm.get_time(current, i) + loc.service_time
                    return_time = new_time + self.dm.get_time(i, self.depot_idx)
                    
                    if (new_demand <= self.vehicle_capacity and
                        return_time <= self.max_route_time and
                        dist < best_dist):
                        
                        # Check time windows if applicable
                        if loc.has_time_window:
                            arrival = route_time + self.dm.get_time(current, i)
                            if arrival > loc.time_window_end:
                                continue
                        
                        best_next = i
                        best_dist = dist
                
                if best_next is None:
                    break
                
                # Add to route
                route.stops.append(best_next)
                visited.add(best_next)
                route_demand += self.dm.locations[best_next].demand
                route_time += self.dm.get_time(current, best_next)
                route_time += self.dm.locations[best_next].service_time
                current = best_next
            
            # Return to depot
            route.stops.append(self.depot_idx)
            route.total_demand = route_demand
            routes.append(route)
            
            # Check if all customers are visited
            if len(visited) == self.dm.size:
                break
        
        # Collect unrouted customers
        for i in range(self.dm.size):
            if i not in visited and i != self.depot_idx:
                unrouted.append(i)
        
        return VRPSolution(routes=routes, unrouted=unrouted)
    
    def _savings_algorithm(self) -> VRPSolution:
        """
        Construct routes using Clarke-Wright Savings algorithm.
        
        The savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
        represents the distance saved by serving i and j in one route
        instead of two separate routes.
        """
        n = self.dm.size
        if n <= 1:
            return VRPSolution(routes=[])
        
        # Calculate savings for all pairs
        savings = []
        for i in range(n):
            if i == self.depot_idx:
                continue
            for j in range(i + 1, n):
                if j == self.depot_idx:
                    continue
                
                s = (self.dm.get_distance(self.depot_idx, i) +
                     self.dm.get_distance(self.depot_idx, j) -
                     self.dm.get_distance(i, j))
                savings.append((s, i, j))
        
        # Sort by savings descending
        savings.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize routes (each customer in own route)
        customer_route = {}  # Maps customer to route index
        routes = []
        
        for i in range(n):
            if i == self.depot_idx:
                continue
            route = Route(
                vehicle_id=f"V{len(routes)+1:03d}",
                stops=[self.depot_idx, i, self.depot_idx],
                total_demand=self.dm.locations[i].demand,
            )
            customer_route[i] = len(routes)
            routes.append(route)
        
        # Merge routes based on savings
        for s, i, j in savings:
            if i not in customer_route or j not in customer_route:
                continue
            
            ri = customer_route[i]
            rj = customer_route[j]
            
            if ri == rj:
                continue  # Already in same route
            
            route_i = routes[ri]
            route_j = routes[rj]
            
            # Check if i and j are at route ends (adjacent to depot)
            i_at_end = (route_i.stops[1] == i or route_i.stops[-2] == i)
            j_at_end = (route_j.stops[1] == j or route_j.stops[-2] == j)
            
            if not (i_at_end and j_at_end):
                continue
            
            # Check capacity constraint
            new_demand = route_i.total_demand + route_j.total_demand
            if new_demand > self.vehicle_capacity:
                continue
            
            # Merge routes
            new_stops = self._merge_routes(route_i.stops, route_j.stops, i, j)
            
            # Check time constraint
            new_time = self.dm.get_total_time(new_stops)
            if new_time > self.max_route_time:
                continue
            
            # Perform merge
            route_i.stops = new_stops
            route_i.total_demand = new_demand
            
            # Update customer mappings
            for c in route_j.stops:
                if c != self.depot_idx:
                    customer_route[c] = ri
            
            # Mark route j as empty
            route_j.stops = [self.depot_idx, self.depot_idx]
            route_j.total_demand = 0
        
        # Filter out empty routes
        routes = [r for r in routes if r.num_stops > 0]
        
        # Renumber vehicle IDs
        for i, r in enumerate(routes):
            r.vehicle_id = f"V{i+1:03d}"
        
        return VRPSolution(routes=routes)
    
    def _merge_routes(
        self,
        route1: List[int],
        route2: List[int],
        i: int,
        j: int,
    ) -> List[int]:
        """Merge two routes at points i and j."""
        # Remove depot from ends
        r1 = route1[1:-1]
        r2 = route2[1:-1]
        
        # Determine orientation
        if r1[-1] == i and r2[0] == j:
            merged = r1 + r2
        elif r1[-1] == i and r2[-1] == j:
            merged = r1 + r2[::-1]
        elif r1[0] == i and r2[0] == j:
            merged = r1[::-1] + r2
        elif r1[0] == i and r2[-1] == j:
            merged = r1[::-1] + r2[::-1]
        else:
            merged = r1 + r2
        
        return [self.depot_idx] + merged + [self.depot_idx]
    
    def _improve_solution(
        self,
        solution: VRPSolution,
        max_iterations: int,
    ) -> VRPSolution:
        """
        Improve solution using local search.
        
        Applies 2-opt and Or-opt moves to reduce total distance.
        """
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for route in solution.routes:
                if route.num_stops < 2:
                    continue
                
                # Try 2-opt
                if self._apply_2opt(route):
                    improved = True
                
                # Try Or-opt
                if self._apply_or_opt(route):
                    improved = True
        
        solution.iterations = iterations
        return solution
    
    def _apply_2opt(self, route: Route) -> bool:
        """
        Apply 2-opt improvement to a route.
        
        2-opt removes two edges and reconnects the tour differently.
        """
        stops = route.stops
        n = len(stops)
        improved = False
        
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):
                # Calculate improvement
                current = (self.dm.get_distance(stops[i-1], stops[i]) +
                          self.dm.get_distance(stops[j], stops[j+1]))
                new = (self.dm.get_distance(stops[i-1], stops[j]) +
                      self.dm.get_distance(stops[i], stops[j+1]))
                
                if new < current - 0.001:  # Small epsilon for float comparison
                    # Reverse segment
                    route.stops = stops[:i] + stops[i:j+1][::-1] + stops[j+1:]
                    stops = route.stops
                    improved = True
        
        return improved
    
    def _apply_or_opt(self, route: Route) -> bool:
        """
        Apply Or-opt improvement to a route.
        
        Or-opt moves a sequence of 1-3 consecutive customers to a new position.
        """
        stops = route.stops
        n = len(stops)
        improved = False
        
        for seq_len in [1, 2, 3]:  # Try different sequence lengths
            for i in range(1, n - seq_len - 1):
                seq = stops[i:i + seq_len]
                
                for j in range(1, n - 1):
                    if abs(j - i) <= seq_len:
                        continue
                    
                    # Calculate current cost
                    current = (
                        self.dm.get_distance(stops[i-1], stops[i]) +
                        self.dm.get_distance(stops[i+seq_len-1], stops[i+seq_len])
                    )
                    
                    # Calculate new cost
                    if j < i:
                        new = (
                            self.dm.get_distance(stops[j-1], seq[0]) +
                            self.dm.get_distance(seq[-1], stops[j])
                        )
                    else:
                        new = (
                            self.dm.get_distance(stops[j], seq[0]) +
                            self.dm.get_distance(seq[-1], stops[j+1])
                        )
                    
                    if new < current - 0.001:
                        # Move sequence
                        new_stops = stops[:i] + stops[i+seq_len:]
                        insert_pos = j if j < i else j - seq_len + 1
                        new_stops = new_stops[:insert_pos] + seq + new_stops[insert_pos:]
                        route.stops = new_stops
                        stops = route.stops
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        return improved
    
    def _calculate_solution_metrics(self, solution: VRPSolution) -> None:
        """Calculate final metrics for the solution."""
        total_distance = 0.0
        total_time = 0.0
        
        for route in solution.routes:
            route.total_distance = self.dm.get_total_distance(route.stops)
            route.total_time = self.dm.get_total_time(route.stops)
            
            # Calculate arrival times
            route.arrival_times = self._calculate_arrival_times(route.stops)
            
            total_distance += route.total_distance
            total_time += route.total_time
        
        solution.total_distance = total_distance
        solution.total_time = total_time
    
    def _calculate_arrival_times(self, stops: List[int]) -> List[float]:
        """Calculate arrival time at each stop."""
        times = [0.0]  # Start at depot
        current_time = 0.0
        
        for i in range(1, len(stops)):
            travel_time = self.dm.get_time(stops[i-1], stops[i])
            current_time += travel_time
            
            # Handle time windows - wait if arriving early
            loc = self.dm.locations[stops[i]]
            if loc.has_time_window and current_time < loc.time_window_start:
                current_time = loc.time_window_start
            
            times.append(current_time)
            
            # Add service time (except at depot)
            if stops[i] != self.depot_idx:
                current_time += loc.service_time
        
        return times
