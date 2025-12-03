"""
Tests for the Vehicle Routing module.
"""

import pytest
from src.routing.distance import Location, DistanceMatrix, haversine_distance, euclidean_distance
from src.routing.vrp_solver import VRPSolver, Route, VRPSolution


class TestLocation:
    """Tests for Location class."""
    
    def test_location_creation(self):
        """Test location creation."""
        loc = Location("LOC-001", "Test Location", 40.7128, -74.0060)
        assert loc.location_id == "LOC-001"
        assert loc.name == "Test Location"
        assert loc.latitude == pytest.approx(40.7128, 0.0001)
        assert loc.longitude == pytest.approx(-74.0060, 0.0001)
    
    def test_location_coordinates(self):
        """Test location coordinates property."""
        loc = Location("L1", "Test", 40.0, -74.0)
        coords = loc.coordinates
        assert coords == (40.0, -74.0)


class TestDistanceCalculations:
    """Tests for distance calculation functions."""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # NYC to Los Angeles (approximately 3944 km)
        nyc = (40.7128, -74.0060)
        la = (34.0522, -118.2437)
        
        distance = haversine_distance(nyc[0], nyc[1], la[0], la[1])
        
        # Should be around 3944 km (allowing some tolerance)
        assert distance == pytest.approx(3944, rel=0.05)
    
    def test_haversine_same_point(self):
        """Test haversine distance between same point is zero."""
        distance = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert distance == 0
    
    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        distance = euclidean_distance(0, 0, 3, 4)
        assert distance == 5.0
    
    def test_euclidean_same_point(self):
        """Test euclidean distance between same point is zero."""
        distance = euclidean_distance(5, 5, 5, 5)
        assert distance == 0


class TestDistanceMatrix:
    """Tests for DistanceMatrix class."""
    
    def test_matrix_creation(self):
        """Test distance matrix creation."""
        locations = [
            Location("A", "Point A", 40.0, -74.0),
            Location("B", "Point B", 40.1, -74.0),
            Location("C", "Point C", 40.0, -74.1),
        ]
        
        matrix = DistanceMatrix.from_locations(locations)
        
        assert matrix is not None
        assert len(matrix.location_ids) == 3
    
    def test_get_distance(self):
        """Test getting distance between two locations."""
        locations = [
            Location("A", "Point A", 40.0, -74.0),
            Location("B", "Point B", 40.1, -74.0),
        ]
        
        matrix = DistanceMatrix.from_locations(locations)
        
        dist_ab = matrix.get_distance("A", "B")
        dist_ba = matrix.get_distance("B", "A")
        
        # Should be symmetric
        assert dist_ab == pytest.approx(dist_ba, 0.001)
        assert dist_ab > 0
    
    def test_same_location_distance(self):
        """Test distance to same location is zero."""
        locations = [
            Location("A", "Point A", 40.0, -74.0),
            Location("B", "Point B", 40.1, -74.0),
        ]
        
        matrix = DistanceMatrix.from_locations(locations)
        
        assert matrix.get_distance("A", "A") == 0
    
    def test_euclidean_method(self):
        """Test matrix creation with euclidean method."""
        locations = [
            Location("A", "Point A", 0, 0),
            Location("B", "Point B", 3, 4),
        ]
        
        matrix = DistanceMatrix.from_locations(locations, method="euclidean")
        
        assert matrix.get_distance("A", "B") == 5.0


class TestRoute:
    """Tests for Route class."""
    
    def test_route_creation(self):
        """Test route creation."""
        route = Route(
            vehicle_id="V1",
            stops=["DEPOT", "A", "B", "DEPOT"],
            total_distance=100.0,
            total_demand=10
        )
        
        assert route.vehicle_id == "V1"
        assert len(route.stops) == 4
        assert route.total_distance == 100.0
        assert route.total_demand == 10
    
    def test_route_num_stops(self):
        """Test counting number of stops (excluding depot)."""
        route = Route(
            vehicle_id="V1",
            stops=["DEPOT", "A", "B", "C", "DEPOT"],
            total_distance=100.0,
            total_demand=15
        )
        
        # 3 customer stops (A, B, C)
        num_customer_stops = len(route.stops) - 2  # Exclude depot at start and end
        assert num_customer_stops == 3


class TestVRPSolver:
    """Tests for VRPSolver class."""
    
    @pytest.fixture
    def sample_locations(self):
        """Create sample locations for testing."""
        return [
            Location("DEPOT", "Distribution Center", 40.7128, -74.0060),
            Location("A", "Customer A", 40.7589, -73.9851),
            Location("B", "Customer B", 40.6782, -73.9442),
            Location("C", "Customer C", 40.7282, -73.7949),
        ]
    
    @pytest.fixture
    def sample_matrix(self, sample_locations):
        """Create sample distance matrix."""
        return DistanceMatrix.from_locations(sample_locations)
    
    def test_solver_creation(self, sample_matrix):
        """Test VRP solver creation."""
        solver = VRPSolver(sample_matrix)
        assert solver is not None
    
    def test_simple_solve(self, sample_matrix):
        """Test solving a simple VRP."""
        solver = VRPSolver(sample_matrix)
        
        demands = {
            "DEPOT": 0,
            "A": 2,
            "B": 3,
            "C": 2,
        }
        
        solution = solver.solve(
            depot_id="DEPOT",
            demands=demands,
            vehicle_capacity=10,
            num_vehicles=1
        )
        
        assert solution is not None
        assert len(solution.routes) >= 1
        assert solution.total_distance > 0
    
    def test_multiple_vehicles(self, sample_matrix):
        """Test solving with multiple vehicles."""
        solver = VRPSolver(sample_matrix)
        
        demands = {
            "DEPOT": 0,
            "A": 5,
            "B": 5,
            "C": 5,
        }
        
        solution = solver.solve(
            depot_id="DEPOT",
            demands=demands,
            vehicle_capacity=10,
            num_vehicles=2
        )
        
        assert len(solution.routes) <= 2
        
        # All customers should be visited
        all_stops = set()
        for route in solution.routes:
            all_stops.update(route.stops)
        
        assert "A" in all_stops
        assert "B" in all_stops
        assert "C" in all_stops
    
    def test_capacity_constraint(self, sample_matrix):
        """Test that vehicle capacity is respected."""
        solver = VRPSolver(sample_matrix)
        
        demands = {
            "DEPOT": 0,
            "A": 10,  # Max capacity
            "B": 10,  # Max capacity
            "C": 10,  # Max capacity
        }
        
        solution = solver.solve(
            depot_id="DEPOT",
            demands=demands,
            vehicle_capacity=10,
            num_vehicles=3
        )
        
        # Each route should respect capacity
        for route in solution.routes:
            assert route.total_demand <= 10
    
    def test_all_customers_served(self, sample_matrix):
        """Test that all customers are served."""
        solver = VRPSolver(sample_matrix)
        
        demands = {
            "DEPOT": 0,
            "A": 2,
            "B": 3,
            "C": 4,
        }
        
        solution = solver.solve(
            depot_id="DEPOT",
            demands=demands,
            vehicle_capacity=10,
            num_vehicles=2
        )
        
        # Count total demand served
        total_served = sum(r.total_demand for r in solution.routes)
        expected_total = sum(d for loc, d in demands.items() if loc != "DEPOT")
        
        assert total_served == expected_total


class TestVRPSolution:
    """Tests for VRPSolution class."""
    
    def test_solution_creation(self):
        """Test solution creation."""
        routes = [
            Route("V1", ["DEPOT", "A", "B", "DEPOT"], 50.0, 5),
            Route("V2", ["DEPOT", "C", "DEPOT"], 30.0, 3),
        ]
        
        solution = VRPSolution(routes=routes, total_distance=80.0)
        
        assert len(solution.routes) == 2
        assert solution.total_distance == 80.0
    
    def test_total_demand_calculation(self):
        """Test total demand calculation across all routes."""
        routes = [
            Route("V1", ["DEPOT", "A", "B", "DEPOT"], 50.0, 5),
            Route("V2", ["DEPOT", "C", "DEPOT"], 30.0, 3),
        ]
        
        solution = VRPSolution(routes=routes, total_distance=80.0)
        
        total_demand = sum(r.total_demand for r in solution.routes)
        assert total_demand == 8


class TestRouteOptimization:
    """Tests for route optimization features."""
    
    def test_route_improvement(self):
        """Test that optimization improves the route."""
        locations = [
            Location("DEPOT", "Depot", 0, 0),
            Location("A", "A", 1, 0),
            Location("B", "B", 2, 0),
            Location("C", "C", 3, 0),
            Location("D", "D", 4, 0),
        ]
        
        matrix = DistanceMatrix.from_locations(locations, method="euclidean")
        solver = VRPSolver(matrix)
        
        demands = {"DEPOT": 0, "A": 1, "B": 1, "C": 1, "D": 1}
        
        solution = solver.solve(
            depot_id="DEPOT",
            demands=demands,
            vehicle_capacity=10,
            num_vehicles=1
        )
        
        # The optimal route for points in a line should visit them in order
        # DEPOT -> A -> B -> C -> D -> DEPOT = 8 units
        assert solution.total_distance == pytest.approx(8.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
