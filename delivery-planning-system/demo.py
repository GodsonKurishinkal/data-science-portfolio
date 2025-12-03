#!/usr/bin/env python3
"""
Delivery Planning System - Interactive Demo
============================================

This demo showcases the key features of the Delivery Planning System:
1. 3D Bin Packing - Optimal loading of packages into delivery vehicles
2. Route Optimization - Finding the best delivery routes
3. Resource Planning - Driver and vehicle assignment
4. Complete Delivery Planning - End-to-end planning workflow
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from packing.box import Box, BoxType
from packing.container import Container, ContainerType
from packing.bin_packer import BinPacker, PackingStrategy, SortingCriterion
from routing.distance import Location, DistanceMatrix
from routing.vrp_solver import VRPSolver
from resources.driver import Driver, DriverPool, DriverSkill, WorkingHours
from resources.scheduler import ResourceScheduler, Shift
from vehicles.vehicle import Vehicle, VehicleType
from vehicles.fleet import Fleet, FleetManager
from planning.delivery_planner import DeliveryPlanner, DeliveryOrder
from utils.visualizer import PackingVisualizer
from utils.metrics import PackingMetrics, RouteMetrics


def demo_bin_packing():
    """Demonstrate the 3D bin packing algorithm."""
    print("\n" + "=" * 60)
    print("📦 3D BIN PACKING DEMONSTRATION")
    print("=" * 60)
    
    # Create a delivery truck container
    print("\n1. Creating a standard delivery truck (6m x 2.5m x 2.7m)")
    truck = Container(
        container_id="TRUCK-001",
        container_type=ContainerType.BOX_TRUCK,
        length=600,  # cm
        width=250,
        height=270,
        max_weight=3000  # kg
    )
    print(f"   Truck capacity: {truck.volume / 1_000_000:.2f} m³")
    
    # Create sample packages with delivery sequence
    print("\n2. Creating packages for delivery...")
    packages = [
        Box("PKG-001", BoxType.SMALL, length=50, width=40, height=30, weight=8, 
            delivery_sequence=5, fragile=False),
        Box("PKG-002", BoxType.MEDIUM, length=80, width=60, height=50, weight=25,
            delivery_sequence=3, fragile=True),
        Box("PKG-003", BoxType.LARGE, length=100, width=80, height=60, weight=40,
            delivery_sequence=1, fragile=False),
        Box("PKG-004", BoxType.SMALL, length=40, width=35, height=25, weight=5,
            delivery_sequence=4, fragile=False),
        Box("PKG-005", BoxType.MEDIUM, length=70, width=55, height=45, weight=20,
            delivery_sequence=2, fragile=False),
        Box("PKG-006", BoxType.XLARGE, length=120, width=90, height=80, weight=60,
            delivery_sequence=6, fragile=True),
        Box("PKG-007", BoxType.SMALL, length=45, width=38, height=28, weight=7,
            delivery_sequence=7, fragile=False),
        Box("PKG-008", BoxType.MEDIUM, length=65, width=50, height=40, weight=15,
            delivery_sequence=8, fragile=False),
    ]
    
    for pkg in packages:
        print(f"   {pkg.box_id}: {pkg.length}x{pkg.width}x{pkg.height} cm, "
              f"{pkg.weight}kg, Seq: {pkg.delivery_sequence}")
    
    # Initialize packer
    print("\n3. Running 3D bin packing algorithm...")
    packer = BinPacker(
        strategy=PackingStrategy.SEQUENCE_AWARE,
        sorting_criterion=SortingCriterion.LIFO_SEQUENCE
    )
    
    result = packer.pack(packages, truck)
    
    # Display results
    print(f"\n4. Packing Results:")
    print(f"   ✅ Boxes packed: {len(result.packed_boxes)}/{len(packages)}")
    print(f"   ❌ Boxes unpacked: {len(result.unpacked_boxes)}")
    print(f"   📊 Volume utilization: {result.volume_utilization:.1f}%")
    print(f"   ⚖️  Weight utilization: {result.weight_utilization:.1f}%")
    
    print("\n5. Packed box positions (LIFO - last delivery loaded first):")
    for box in sorted(result.packed_boxes, key=lambda b: b.delivery_sequence, reverse=True):
        pos = box.position
        print(f"   {box.box_id} @ position ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f}) - "
              f"Seq: {box.delivery_sequence}")
    
    # Calculate metrics
    metrics = PackingMetrics.calculate(result.packed_boxes, truck)
    print(f"\n6. Packing Metrics:")
    print(f"   Space efficiency: {metrics.space_efficiency:.1f}%")
    print(f"   Weight efficiency: {metrics.weight_efficiency:.1f}%")
    print(f"   Stacking ratio: {metrics.stacking_ratio:.2f}")
    
    return result


def demo_route_optimization():
    """Demonstrate the vehicle routing optimization."""
    print("\n" + "=" * 60)
    print("🗺️  ROUTE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create delivery locations
    print("\n1. Setting up delivery locations...")
    locations = [
        Location("DEPOT", "Distribution Center", 40.7128, -74.0060),  # NYC
        Location("STOP-1", "Customer A - Manhattan", 40.7589, -73.9851),
        Location("STOP-2", "Customer B - Brooklyn", 40.6782, -73.9442),
        Location("STOP-3", "Customer C - Queens", 40.7282, -73.7949),
        Location("STOP-4", "Customer D - Bronx", 40.8448, -73.8648),
        Location("STOP-5", "Customer E - Staten Island", 40.5795, -74.1502),
        Location("STOP-6", "Customer F - Jersey City", 40.7178, -74.0431),
    ]
    
    for loc in locations:
        print(f"   {loc.location_id}: {loc.name}")
    
    # Create distance matrix
    print("\n2. Calculating distance matrix...")
    distance_matrix = DistanceMatrix.from_locations(locations, method="haversine")
    
    # Set demands (package counts per location)
    demands = {
        "DEPOT": 0,
        "STOP-1": 3,
        "STOP-2": 5,
        "STOP-3": 2,
        "STOP-4": 4,
        "STOP-5": 3,
        "STOP-6": 4,
    }
    
    # Solve VRP
    print("\n3. Solving Vehicle Routing Problem...")
    solver = VRPSolver(distance_matrix)
    solution = solver.solve(
        depot_id="DEPOT",
        demands=demands,
        vehicle_capacity=10,
        num_vehicles=2
    )
    
    # Display results
    print(f"\n4. Routing Results:")
    print(f"   Total distance: {solution.total_distance:.2f} km")
    print(f"   Vehicles used: {len(solution.routes)}")
    
    for i, route in enumerate(solution.routes, 1):
        print(f"\n   🚚 Vehicle {i}:")
        print(f"      Route: {' → '.join(route.stops)}")
        print(f"      Distance: {route.total_distance:.2f} km")
        print(f"      Load: {route.total_demand} packages")
    
    # Calculate route metrics
    for i, route in enumerate(solution.routes, 1):
        metrics = RouteMetrics(
            total_distance=route.total_distance,
            total_time=route.total_distance / 30,  # Assume 30 km/h avg speed
            num_stops=len(route.stops) - 2,  # Exclude depot
            avg_distance_between_stops=route.total_distance / max(1, len(route.stops) - 1)
        )
        print(f"\n   Vehicle {i} Metrics:")
        print(f"      Estimated time: {metrics.total_time:.1f} hours")
        print(f"      Avg between stops: {metrics.avg_distance_between_stops:.2f} km")
    
    return solution


def demo_resource_planning():
    """Demonstrate driver and resource planning."""
    print("\n" + "=" * 60)
    print("👷 RESOURCE PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create drivers
    print("\n1. Setting up driver pool...")
    drivers = [
        Driver(
            driver_id="DRV-001",
            name="John Smith",
            skills=[DriverSkill.STANDARD, DriverSkill.FRAGILE_HANDLING],
            working_hours=WorkingHours(start_time=8, end_time=17),
            max_daily_hours=8
        ),
        Driver(
            driver_id="DRV-002",
            name="Jane Doe",
            skills=[DriverSkill.STANDARD, DriverSkill.HEAVY_GOODS, DriverSkill.HAZMAT],
            working_hours=WorkingHours(start_time=6, end_time=14),
            max_daily_hours=8
        ),
        Driver(
            driver_id="DRV-003",
            name="Mike Johnson",
            skills=[DriverSkill.STANDARD, DriverSkill.REFRIGERATED],
            working_hours=WorkingHours(start_time=10, end_time=19),
            max_daily_hours=9
        ),
    ]
    
    pool = DriverPool()
    for driver in drivers:
        pool.add_driver(driver)
        skills_str = ", ".join(s.value for s in driver.skills)
        print(f"   {driver.driver_id}: {driver.name} - Skills: {skills_str}")
    
    # Create shifts
    print("\n2. Defining shifts...")
    today = datetime.now().date()
    shifts = [
        Shift(
            shift_id="SHIFT-AM",
            date=today,
            start_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=6),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=14),
            required_skills=[DriverSkill.STANDARD]
        ),
        Shift(
            shift_id="SHIFT-PM",
            date=today,
            start_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=14),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=22),
            required_skills=[DriverSkill.STANDARD]
        ),
        Shift(
            shift_id="SHIFT-HEAVY",
            date=today,
            start_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=8),
            end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=16),
            required_skills=[DriverSkill.HEAVY_GOODS]
        ),
    ]
    
    for shift in shifts:
        print(f"   {shift.shift_id}: {shift.start_time.strftime('%H:%M')} - "
              f"{shift.end_time.strftime('%H:%M')}")
    
    # Schedule resources
    print("\n3. Scheduling drivers to shifts...")
    scheduler = ResourceScheduler(pool)
    assignments = scheduler.assign_shifts(shifts)
    
    print(f"\n4. Assignment Results:")
    for assignment in assignments:
        if assignment.driver_id:
            driver = pool.get_driver(assignment.driver_id)
            print(f"   {assignment.shift_id} → {driver.name}")
        else:
            print(f"   {assignment.shift_id} → ⚠️ Unassigned")
    
    return assignments


def demo_vehicle_planning():
    """Demonstrate fleet and vehicle planning."""
    print("\n" + "=" * 60)
    print("🚛 VEHICLE PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create fleet
    print("\n1. Setting up vehicle fleet...")
    vehicles = [
        Vehicle("VEH-001", VehicleType.SMALL_VAN, 
                capacity_weight=1000, capacity_volume=8_000_000),
        Vehicle("VEH-002", VehicleType.LARGE_VAN,
                capacity_weight=2000, capacity_volume=15_000_000),
        Vehicle("VEH-003", VehicleType.BOX_TRUCK,
                capacity_weight=5000, capacity_volume=40_000_000),
        Vehicle("VEH-004", VehicleType.BOX_TRUCK,
                capacity_weight=5000, capacity_volume=40_000_000),
        Vehicle("VEH-005", VehicleType.SEMI_TRAILER,
                capacity_weight=20000, capacity_volume=80_000_000),
    ]
    
    fleet = Fleet()
    for vehicle in vehicles:
        fleet.add_vehicle(vehicle)
        print(f"   {vehicle.vehicle_id}: {vehicle.vehicle_type.value} - "
              f"{vehicle.capacity_weight}kg, {vehicle.capacity_volume/1_000_000:.0f}m³")
    
    # Fleet manager for allocation
    print("\n2. Creating fleet allocation plan...")
    manager = FleetManager(fleet)
    
    # Sample delivery requirements
    requirements = [
        {"weight": 800, "volume": 5_000_000, "priority": "high"},
        {"weight": 1500, "volume": 12_000_000, "priority": "medium"},
        {"weight": 4000, "volume": 35_000_000, "priority": "high"},
        {"weight": 15000, "volume": 70_000_000, "priority": "low"},
    ]
    
    print("\n3. Allocating vehicles to deliveries...")
    for i, req in enumerate(requirements, 1):
        vehicle = manager.allocate_vehicle(req["weight"], req["volume"])
        if vehicle:
            print(f"   Delivery {i} ({req['weight']}kg, {req['volume']/1_000_000:.0f}m³) "
                  f"→ {vehicle.vehicle_id} ({vehicle.vehicle_type.value})")
            manager.reserve_vehicle(vehicle.vehicle_id)
        else:
            print(f"   Delivery {i} → ⚠️ No suitable vehicle available")
    
    # Fleet status
    print("\n4. Fleet Status:")
    available = manager.get_available_vehicles()
    print(f"   Available vehicles: {len(available)}/{len(fleet.vehicles)}")
    for v in available:
        print(f"      {v.vehicle_id}: {v.vehicle_type.value}")
    
    return fleet


def demo_complete_planning():
    """Demonstrate complete end-to-end delivery planning."""
    print("\n" + "=" * 60)
    print("📋 COMPLETE DELIVERY PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create delivery orders
    print("\n1. Creating delivery orders...")
    orders = [
        DeliveryOrder(
            order_id="ORD-001",
            customer_id="CUST-A",
            location=Location("LOC-A", "Customer A", 40.7589, -73.9851),
            packages=[
                Box("PKG-A1", BoxType.MEDIUM, 60, 45, 35, 15),
                Box("PKG-A2", BoxType.SMALL, 35, 30, 25, 8),
            ],
            time_window_start=datetime.now() + timedelta(hours=2),
            time_window_end=datetime.now() + timedelta(hours=4),
            priority=1
        ),
        DeliveryOrder(
            order_id="ORD-002",
            customer_id="CUST-B",
            location=Location("LOC-B", "Customer B", 40.6782, -73.9442),
            packages=[
                Box("PKG-B1", BoxType.LARGE, 100, 80, 60, 45),
            ],
            time_window_start=datetime.now() + timedelta(hours=3),
            time_window_end=datetime.now() + timedelta(hours=6),
            priority=2
        ),
        DeliveryOrder(
            order_id="ORD-003",
            customer_id="CUST-C",
            location=Location("LOC-C", "Customer C", 40.7282, -73.7949),
            packages=[
                Box("PKG-C1", BoxType.SMALL, 40, 35, 30, 10),
                Box("PKG-C2", BoxType.SMALL, 38, 32, 28, 9),
                Box("PKG-C3", BoxType.MEDIUM, 55, 45, 35, 18),
            ],
            time_window_start=datetime.now() + timedelta(hours=1),
            time_window_end=datetime.now() + timedelta(hours=5),
            priority=1
        ),
    ]
    
    for order in orders:
        print(f"   {order.order_id}: {len(order.packages)} packages to {order.customer_id}")
    
    # Initialize planner
    print("\n2. Initializing delivery planner...")
    depot = Location("DEPOT", "Distribution Center", 40.7128, -74.0060)
    
    # Create a simple fleet
    fleet = Fleet()
    fleet.add_vehicle(Vehicle("TRUCK-1", VehicleType.BOX_TRUCK, 3000, 40_000_000))
    fleet.add_vehicle(Vehicle("VAN-1", VehicleType.LARGE_VAN, 2000, 15_000_000))
    
    # Create driver pool
    pool = DriverPool()
    pool.add_driver(Driver("DRV-1", "Driver One", [DriverSkill.STANDARD]))
    pool.add_driver(Driver("DRV-2", "Driver Two", [DriverSkill.STANDARD]))
    
    planner = DeliveryPlanner(depot, fleet, pool)
    
    # Generate plan
    print("\n3. Generating delivery plan...")
    plan = planner.plan_deliveries(orders)
    
    # Display results
    print(f"\n4. Planning Results:")
    print(f"   ✅ Orders planned: {len(plan.assigned_orders)}/{len(orders)}")
    print(f"   🚚 Vehicles used: {len(plan.vehicle_assignments)}")
    print(f"   👷 Drivers assigned: {len(plan.driver_assignments)}")
    
    for vehicle_id, route in plan.vehicle_assignments.items():
        driver_id = plan.driver_assignments.get(vehicle_id, "Unassigned")
        print(f"\n   Vehicle {vehicle_id} (Driver: {driver_id}):")
        print(f"      Route: {' → '.join(route.stops)}")
        print(f"      Distance: {route.total_distance:.2f} km")
        print(f"      Est. time: {route.total_distance / 30:.1f} hours")
    
    print(f"\n5. Summary:")
    print(f"   Total distance: {plan.total_distance:.2f} km")
    print(f"   Total packages: {plan.total_packages}")
    print(f"   Estimated completion: {plan.estimated_completion}")
    
    return plan


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("🚚 DELIVERY PLANNING SYSTEM - INTERACTIVE DEMO 🚚")
    print("=" * 60)
    print("\nThis demo showcases the complete delivery planning system")
    print("including 3D bin packing, route optimization, and resource planning.\n")
    
    demos = [
        ("1", "3D Bin Packing", demo_bin_packing),
        ("2", "Route Optimization", demo_route_optimization),
        ("3", "Resource Planning", demo_resource_planning),
        ("4", "Vehicle Planning", demo_vehicle_planning),
        ("5", "Complete Planning", demo_complete_planning),
        ("A", "Run All Demos", None),
    ]
    
    print("Available Demonstrations:")
    for key, name, _ in demos:
        print(f"   [{key}] {name}")
    print("   [Q] Quit\n")
    
    while True:
        choice = input("Select demo (1-5, A, or Q): ").strip().upper()
        
        if choice == "Q":
            print("\nThank you for exploring the Delivery Planning System!")
            break
        elif choice == "A":
            for key, name, func in demos[:-1]:  # Exclude "Run All"
                func()
            print("\n" + "=" * 60)
            print("✅ All demonstrations completed!")
            print("=" * 60)
        elif choice in [d[0] for d in demos[:-1]]:
            for key, name, func in demos[:-1]:
                if key == choice:
                    func()
                    break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
