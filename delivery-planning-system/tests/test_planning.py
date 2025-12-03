"""
Tests for the Delivery Planning module.
"""

import pytest
from datetime import datetime, timedelta

from src.packing.box import Box, BoxType
from src.routing.distance import Location
from src.vehicles.vehicle import Vehicle, VehicleType
from src.vehicles.fleet import Fleet
from src.resources.driver import Driver, DriverPool, DriverSkill
from src.planning.delivery_planner import DeliveryPlanner, DeliveryOrder, DeliveryPlan


class TestDeliveryOrder:
    """Tests for DeliveryOrder class."""
    
    def test_order_creation(self):
        """Test delivery order creation."""
        packages = [
            Box("P1", BoxType.SMALL, 30, 20, 15, 5),
            Box("P2", BoxType.MEDIUM, 50, 40, 30, 10),
        ]
        location = Location("L1", "Customer 1", 40.0, -74.0)
        
        order = DeliveryOrder(
            order_id="ORD-001",
            customer_id="CUST-001",
            location=location,
            packages=packages,
            priority=1
        )
        
        assert order.order_id == "ORD-001"
        assert len(order.packages) == 2
        assert order.priority == 1
    
    def test_order_with_time_window(self):
        """Test order with time window."""
        location = Location("L1", "Customer 1", 40.0, -74.0)
        now = datetime.now()
        
        order = DeliveryOrder(
            order_id="ORD-001",
            customer_id="CUST-001",
            location=location,
            packages=[Box("P1", BoxType.SMALL, 30, 20, 15, 5)],
            time_window_start=now + timedelta(hours=2),
            time_window_end=now + timedelta(hours=4),
        )
        
        assert order.time_window_start is not None
        assert order.time_window_end is not None
    
    def test_order_total_weight(self):
        """Test calculating total weight of order."""
        packages = [
            Box("P1", BoxType.SMALL, 30, 20, 15, 5),
            Box("P2", BoxType.MEDIUM, 50, 40, 30, 10),
        ]
        location = Location("L1", "Customer 1", 40.0, -74.0)
        
        order = DeliveryOrder(
            order_id="ORD-001",
            customer_id="CUST-001",
            location=location,
            packages=packages,
        )
        
        total_weight = sum(p.weight for p in order.packages)
        assert total_weight == 15
    
    def test_order_total_volume(self):
        """Test calculating total volume of order."""
        packages = [
            Box("P1", BoxType.SMALL, 30, 20, 15, 5),  # 9000
            Box("P2", BoxType.SMALL, 20, 20, 20, 5),  # 8000
        ]
        location = Location("L1", "Customer 1", 40.0, -74.0)
        
        order = DeliveryOrder(
            order_id="ORD-001",
            customer_id="CUST-001",
            location=location,
            packages=packages,
        )
        
        total_volume = sum(p.volume for p in order.packages)
        assert total_volume == 17000


class TestDeliveryPlanner:
    """Tests for DeliveryPlanner class."""
    
    @pytest.fixture
    def sample_fleet(self):
        """Create sample fleet for testing."""
        fleet = Fleet()
        fleet.add_vehicle(Vehicle("V1", VehicleType.BOX_TRUCK, 3000, 40_000_000))
        fleet.add_vehicle(Vehicle("V2", VehicleType.LARGE_VAN, 2000, 15_000_000))
        return fleet
    
    @pytest.fixture
    def sample_drivers(self):
        """Create sample driver pool for testing."""
        pool = DriverPool()
        pool.add_driver(Driver("D1", "Driver 1", [DriverSkill.STANDARD]))
        pool.add_driver(Driver("D2", "Driver 2", [DriverSkill.STANDARD]))
        return pool
    
    @pytest.fixture
    def sample_depot(self):
        """Create sample depot location."""
        return Location("DEPOT", "Distribution Center", 40.7128, -74.0060)
    
    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        return [
            DeliveryOrder(
                order_id="O1",
                customer_id="C1",
                location=Location("L1", "Customer 1", 40.7589, -73.9851),
                packages=[Box("P1", BoxType.SMALL, 30, 20, 15, 5)],
                priority=1
            ),
            DeliveryOrder(
                order_id="O2",
                customer_id="C2",
                location=Location("L2", "Customer 2", 40.6782, -73.9442),
                packages=[Box("P2", BoxType.MEDIUM, 50, 40, 30, 15)],
                priority=2
            ),
        ]
    
    def test_planner_creation(self, sample_depot, sample_fleet, sample_drivers):
        """Test planner creation."""
        planner = DeliveryPlanner(sample_depot, sample_fleet, sample_drivers)
        assert planner is not None
    
    def test_plan_deliveries(self, sample_depot, sample_fleet, sample_drivers, sample_orders):
        """Test basic delivery planning."""
        planner = DeliveryPlanner(sample_depot, sample_fleet, sample_drivers)
        
        plan = planner.plan_deliveries(sample_orders)
        
        assert plan is not None
        assert len(plan.assigned_orders) > 0
    
    def test_plan_with_vehicle_assignment(self, sample_depot, sample_fleet, sample_drivers, sample_orders):
        """Test that vehicles are assigned."""
        planner = DeliveryPlanner(sample_depot, sample_fleet, sample_drivers)
        
        plan = planner.plan_deliveries(sample_orders)
        
        assert len(plan.vehicle_assignments) > 0
    
    def test_plan_with_driver_assignment(self, sample_depot, sample_fleet, sample_drivers, sample_orders):
        """Test that drivers are assigned."""
        planner = DeliveryPlanner(sample_depot, sample_fleet, sample_drivers)
        
        plan = planner.plan_deliveries(sample_orders)
        
        assert len(plan.driver_assignments) > 0
    
    def test_empty_orders(self, sample_depot, sample_fleet, sample_drivers):
        """Test planning with no orders."""
        planner = DeliveryPlanner(sample_depot, sample_fleet, sample_drivers)
        
        plan = planner.plan_deliveries([])
        
        assert len(plan.assigned_orders) == 0
        assert plan.total_distance == 0


class TestDeliveryPlan:
    """Tests for DeliveryPlan class."""
    
    def test_plan_creation(self):
        """Test delivery plan creation."""
        plan = DeliveryPlan(
            assigned_orders=["O1", "O2"],
            vehicle_assignments={"V1": None},
            driver_assignments={"V1": "D1"},
            total_distance=100.0,
            total_packages=5,
            estimated_completion=datetime.now() + timedelta(hours=4)
        )
        
        assert len(plan.assigned_orders) == 2
        assert plan.total_distance == 100.0
        assert plan.total_packages == 5
    
    def test_plan_summary(self):
        """Test plan summary generation."""
        plan = DeliveryPlan(
            assigned_orders=["O1", "O2", "O3"],
            vehicle_assignments={"V1": None, "V2": None},
            driver_assignments={"V1": "D1", "V2": "D2"},
            total_distance=150.0,
            total_packages=10,
            estimated_completion=datetime.now() + timedelta(hours=6)
        )
        
        assert len(plan.assigned_orders) == 3
        assert len(plan.vehicle_assignments) == 2
        assert len(plan.driver_assignments) == 2


class TestPlanningIntegration:
    """Integration tests for delivery planning."""
    
    def test_full_planning_workflow(self):
        """Test complete planning workflow."""
        # Setup
        depot = Location("DEPOT", "Depot", 40.7128, -74.0060)
        
        fleet = Fleet()
        fleet.add_vehicle(Vehicle("V1", VehicleType.BOX_TRUCK, 5000, 40_000_000))
        
        pool = DriverPool()
        pool.add_driver(Driver("D1", "John", [DriverSkill.STANDARD]))
        
        orders = [
            DeliveryOrder(
                order_id="ORD-001",
                customer_id="CUST-A",
                location=Location("LA", "Customer A", 40.7589, -73.9851),
                packages=[
                    Box("P1", BoxType.MEDIUM, 50, 40, 30, 15),
                    Box("P2", BoxType.SMALL, 30, 25, 20, 5),
                ],
                priority=1
            ),
            DeliveryOrder(
                order_id="ORD-002",
                customer_id="CUST-B",
                location=Location("LB", "Customer B", 40.6782, -73.9442),
                packages=[
                    Box("P3", BoxType.LARGE, 80, 60, 50, 30),
                ],
                priority=2
            ),
        ]
        
        # Plan
        planner = DeliveryPlanner(depot, fleet, pool)
        plan = planner.plan_deliveries(orders)
        
        # Verify
        assert len(plan.assigned_orders) == 2
        assert plan.total_packages == 3
        assert plan.total_distance > 0
    
    def test_priority_ordering(self):
        """Test that high priority orders are handled."""
        depot = Location("DEPOT", "Depot", 40.7128, -74.0060)
        
        fleet = Fleet()
        fleet.add_vehicle(Vehicle("V1", VehicleType.BOX_TRUCK, 5000, 40_000_000))
        
        pool = DriverPool()
        pool.add_driver(Driver("D1", "John", [DriverSkill.STANDARD]))
        
        orders = [
            DeliveryOrder(
                order_id="LOW",
                customer_id="C1",
                location=Location("L1", "Far", 41.0, -74.5),
                packages=[Box("P1", BoxType.SMALL, 20, 20, 20, 5)],
                priority=3  # Low priority
            ),
            DeliveryOrder(
                order_id="HIGH",
                customer_id="C2",
                location=Location("L2", "Close", 40.72, -74.01),
                packages=[Box("P2", BoxType.SMALL, 20, 20, 20, 5)],
                priority=1  # High priority
            ),
        ]
        
        planner = DeliveryPlanner(depot, fleet, pool)
        plan = planner.plan_deliveries(orders)
        
        # Both should be assigned
        assert "HIGH" in plan.assigned_orders
        assert "LOW" in plan.assigned_orders


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
