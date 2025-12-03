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


def demo_bin_packing():
    """Demonstrate the 3D bin packing algorithm."""
    print("\n" + "=" * 60)
    print("📦 3D BIN PACKING DEMONSTRATION")
    print("=" * 60)
    
    # Create a delivery truck container
    print("\n1. Creating a standard delivery truck (6m x 2.5m x 2.7m)")
    truck = Container(
        id="TRUCK-001",
        container_type=ContainerType.LARGE_TRUCK,
        length=600,  # cm
        width=250,
        height=270,
        max_weight=3000  # kg
    )
    print(f"   Truck capacity: {truck.volume / 1_000_000:.2f} m³")
    
    # Create sample packages with delivery sequence
    print("\n2. Creating packages for delivery...")
    packages = [
        Box(id="PKG-001", length=50, width=40, height=30, weight=8, 
            sequence=5, box_type=BoxType.STANDARD),
        Box(id="PKG-002", length=80, width=60, height=50, weight=25,
            sequence=3, box_type=BoxType.FRAGILE),
        Box(id="PKG-003", length=100, width=80, height=60, weight=40,
            sequence=1, box_type=BoxType.STANDARD),
        Box(id="PKG-004", length=40, width=35, height=25, weight=5,
            sequence=4, box_type=BoxType.STANDARD),
        Box(id="PKG-005", length=70, width=55, height=45, weight=20,
            sequence=2, box_type=BoxType.STANDARD),
        Box(id="PKG-006", length=120, width=90, height=80, weight=60,
            sequence=6, box_type=BoxType.FRAGILE),
        Box(id="PKG-007", length=45, width=38, height=28, weight=7,
            sequence=7, box_type=BoxType.STANDARD),
        Box(id="PKG-008", length=65, width=50, height=40, weight=15,
            sequence=8, box_type=BoxType.STANDARD),
    ]
    
    for pkg in packages:
        print(f"   {pkg.id}: {pkg.length}x{pkg.width}x{pkg.height} cm, "
              f"{pkg.weight}kg, Seq: {pkg.sequence}")
    
    # Initialize packer
    print("\n3. Running 3D bin packing algorithm...")
    packer = BinPacker(
        strategy=PackingStrategy.SEQUENCE_AWARE,
        sorting=SortingCriterion.SEQUENCE_ASC
    )
    
    result = packer.pack(packages, truck)
    
    # Display results
    print("\n4. Packing Results:")
    print(f"   ✅ Boxes packed: {result.num_packed}/{len(packages)}")
    print(f"   ❌ Boxes unpacked: {result.num_unpacked}")
    print(f"   📊 Volume utilization: {result.utilization:.1f}%")
    print(f"   ⚖️  Weight utilization: {result.weight_utilization:.1f}%")
    
    print("\n5. Packed box positions (LIFO - last delivery loaded first):")
    for box in sorted(result.packed_boxes, key=lambda b: b.sequence, reverse=True):
        pos = box.position
        print(f"   {box.id} @ position ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f}) - "
              f"Seq: {box.sequence}")
    
    return result


def demo_route_optimization():
    """Demonstrate the vehicle routing optimization."""
    print("\n" + "=" * 60)
    print("🗺️  ROUTE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create depot and delivery locations
    print("\n1. Setting up depot and delivery locations...")
    depot = Location(id="DEPOT", name="Warehouse", latitude=40.7128, longitude=-74.0060)
    
    locations = [
        depot,
        Location(id="C1", name="Customer 1", latitude=40.7580, longitude=-73.9855),  # Midtown
        Location(id="C2", name="Customer 2", latitude=40.7489, longitude=-73.9680),  # Grand Central
        Location(id="C3", name="Customer 3", latitude=40.7614, longitude=-73.9776),  # Rockefeller Center
        Location(id="C4", name="Customer 4", latitude=40.7484, longitude=-73.9857),  # Empire State
        Location(id="C5", name="Customer 5", latitude=40.7527, longitude=-73.9772),  # Bryant Park
    ]
    
    for loc in locations:
        print(f"   {loc.id}: {loc.name} ({loc.latitude:.4f}, {loc.longitude:.4f})")
    
    # Create distance matrix
    print("\n2. Creating distance matrix...")
    distance_matrix = DistanceMatrix.from_locations(locations)
    print(f"   Matrix size: {len(locations)}x{len(locations)}")
    
    # Solve VRP
    print("\n3. Solving Vehicle Routing Problem...")
    solver = VRPSolver(
        distance_matrix=distance_matrix,
        depot_index=0,
        num_vehicles=2
    )
    
    solution = solver.solve()
    
    # Display results
    print("\n4. Routing Results:")
    print(f"   Total distance: {solution.total_distance:.2f} km")
    print(f"   Vehicles used: {solution.num_vehicles_used}/{2}")
    
    print("\n5. Routes:")
    for i, route in enumerate(solution.routes):
        if not route.is_empty():
            stops = " → ".join([locations[idx].name for idx in route.stops])
            print(f"   Vehicle {i+1}: {stops}")
            print(f"      Distance: {route.distance:.2f} km")
    
    return solution


def demo_resource_planning():
    """Demonstrate driver and resource planning."""
    print("\n" + "=" * 60)
    print("👥 RESOURCE PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create driver pool
    print("\n1. Creating driver pool...")
    drivers = [
        Driver(
            id="DRV-001",
            name="John Smith",
            skill_level=DriverSkill.SENIOR,
            working_hours=WorkingHours(start_time="06:00", end_time="14:00"),
            max_driving_hours=8.0,
            vehicle_types=[VehicleType.MEDIUM_TRUCK, VehicleType.LARGE_TRUCK]
        ),
        Driver(
            id="DRV-002", 
            name="Jane Doe",
            skill_level=DriverSkill.STANDARD,
            working_hours=WorkingHours(start_time="08:00", end_time="18:00"),
            max_driving_hours=10.0,
            vehicle_types=[VehicleType.SMALL_VAN, VehicleType.MEDIUM_TRUCK]
        ),
        Driver(
            id="DRV-003",
            name="Bob Wilson",
            skill_level=DriverSkill.JUNIOR,
            working_hours=WorkingHours(start_time="14:00", end_time="22:00"),
            max_driving_hours=8.0,
            vehicle_types=[VehicleType.SMALL_VAN]
        ),
    ]
    
    pool = DriverPool(drivers=drivers)
    
    for drv in drivers:
        print(f"   {drv.id}: {drv.name} ({drv.skill_level.value})")
        print(f"      Hours: {drv.working_hours.start_time}-{drv.working_hours.end_time}")
    
    # Create scheduler
    print("\n2. Setting up resource scheduler...")
    scheduler = ResourceScheduler(driver_pool=pool)
    
    # Create shifts
    print("\n3. Creating shifts for tomorrow...")
    tomorrow = datetime.now().date() + timedelta(days=1)
    
    shifts = [
        Shift(
            id="SHIFT-001",
            date=tomorrow,
            start_time="06:00",
            end_time="14:00",
            required_skill=DriverSkill.SENIOR,
            vehicle_type=VehicleType.LARGE_TRUCK
        ),
        Shift(
            id="SHIFT-002",
            date=tomorrow,
            start_time="08:00",
            end_time="18:00",
            required_skill=DriverSkill.STANDARD,
            vehicle_type=VehicleType.MEDIUM_TRUCK
        ),
    ]
    
    for shift in shifts:
        print(f"   {shift.id}: {shift.start_time}-{shift.end_time} "
              f"({shift.required_skill.value}, {shift.vehicle_type.value})")
    
    # Assign drivers
    print("\n4. Assignment Results:")
    assignments = scheduler.assign_shifts(shifts)
    for assignment in assignments:
        print(f"   {assignment.shift_id}: Assigned to {assignment.driver_id}")
    
    return assignments


def demo_complete_planning():
    """Demonstrate the complete delivery planning workflow."""
    print("\n" + "=" * 60)
    print("🚚 COMPLETE DELIVERY PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Setup fleet
    print("\n1. Setting up fleet...")
    vehicles = [
        Vehicle.from_type(VehicleType.LARGE_TRUCK, vehicle_id="TRK-001"),
        Vehicle.from_type(VehicleType.MEDIUM_TRUCK, vehicle_id="TRK-002"),
    ]
    fleet = Fleet(vehicles=vehicles)
    
    for v in vehicles:
        print(f"   {v.id}: {v.vehicle_type.value}")
    
    # Setup drivers
    print("\n2. Setting up driver pool...")
    drivers = [
        Driver(id="DRV-001", name="Driver 1", skill_level=DriverSkill.SENIOR),
        Driver(id="DRV-002", name="Driver 2", skill_level=DriverSkill.STANDARD),
    ]
    driver_pool = DriverPool(drivers=drivers)
    
    # Setup depot
    depot = Location(id="DEPOT", name="Warehouse", latitude=40.7128, longitude=-74.0060)
    
    # Create delivery orders
    print("\n3. Creating delivery orders...")
    orders = [
        DeliveryOrder(
            id="ORD-001",
            customer_id="CUST-001",
            destination=Location(id="LOC-001", latitude=40.7580, longitude=-73.9855),
            packages=[
                Box(id="PKG-001", length=50, width=40, height=30, weight=10, sequence=1),
                Box(id="PKG-002", length=30, width=25, height=20, weight=5, sequence=1),
            ],
            priority=1
        ),
        DeliveryOrder(
            id="ORD-002",
            customer_id="CUST-002",
            destination=Location(id="LOC-002", latitude=40.7489, longitude=-73.9680),
            packages=[
                Box(id="PKG-003", length=80, width=60, height=50, weight=25, sequence=2),
            ],
            priority=2
        ),
        DeliveryOrder(
            id="ORD-003",
            customer_id="CUST-003",
            destination=Location(id="LOC-003", latitude=40.7614, longitude=-73.9776),
            packages=[
                Box(id="PKG-004", length=40, width=35, height=25, weight=8, sequence=3),
                Box(id="PKG-005", length=45, width=40, height=30, weight=12, sequence=3),
            ],
            priority=1
        ),
    ]
    
    for order in orders:
        print(f"   {order.id}: {order.num_packages} packages, "
              f"{order.total_weight:.1f}kg, Priority: {order.priority}")
    
    # Initialize planner
    planner = DeliveryPlanner(
        fleet=fleet,
        driver_pool=driver_pool,
        depot_location=depot
    )
    
    # Generate delivery plan
    print("\n4. Planning Results:")
    try:
        result = planner.plan_deliveries(orders)
        
        print(f"   ✅ Orders assigned: {len(orders) - len(result.unassigned_orders)}/{len(orders)}")
        print(f"   🚛 Vehicles used: {result.vehicles_used}")
        print(f"   📏 Total distance: {result.total_distance:.2f} km")
        print(f"   ⏱️  Total time: {result.total_time:.1f} minutes")
        
        print("\n5. Summary:")
        print(f"   Planning completed: {result.is_complete}")
        print(f"   Computation time: {result.computation_time:.2f}s")
        
        return result
    except Exception as e:
        print(f"   ⚠️  Planning error: {str(e)}")
        return None


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("🚚 DELIVERY PLANNING SYSTEM - INTERACTIVE DEMO")
    print("=" * 60)
    print("\nThis demo showcases the key components of the system:")
    print("1. 3D Bin Packing Algorithm")
    print("2. Vehicle Route Optimization")
    print("3. Driver/Resource Planning")
    print("4. Complete Delivery Planning")
    
    while True:
        print("\n" + "-" * 40)
        print("Select a demo to run:")
        print("  1. 3D Bin Packing")
        print("  2. Route Optimization")
        print("  3. Resource Planning")
        print("  4. Complete Planning")
        print("  5. Run All Demos")
        print("  0. Exit")
        
        try:
            choice = input("\nEnter choice (0-5): ").strip()
        except EOFError:
            # Non-interactive mode - run all demos
            choice = "5"
        
        if choice == "0":
            print("\nThank you for exploring the Delivery Planning System!")
            break
        elif choice == "1":
            demo_bin_packing()
        elif choice == "2":
            demo_route_optimization()
        elif choice == "3":
            demo_resource_planning()
        elif choice == "4":
            demo_complete_planning()
        elif choice == "5":
            demo_bin_packing()
            demo_route_optimization()
            demo_resource_planning()
            demo_complete_planning()
            print("\n" + "=" * 60)
            print("All demos completed!")
            print("=" * 60)
            break
        else:
            print("Invalid choice. Please enter 0-5.")


if __name__ == "__main__":
    main()
