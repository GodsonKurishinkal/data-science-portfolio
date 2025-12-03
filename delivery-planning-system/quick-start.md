# Delivery Planning System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Setup Environment

```bash
# Navigate to project directory
cd delivery-planning-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Run the Interactive Demo

```bash
python demo.py
```

This launches an interactive demo showcasing:
- 📦 3D Bin Packing
- 🗺️ Route Optimization
- 👷 Resource Planning
- 🚛 Vehicle Planning
- 📋 Complete Delivery Planning

### 3. Open the 3D Visualization

Open `docs/3d_visualization.html` in your browser for an interactive truck loading simulation:

```bash
# macOS
open docs/3d_visualization.html

# Linux
xdg-open docs/3d_visualization.html

# Windows
start docs/3d_visualization.html
```

**Features:**
- Add packages with custom dimensions
- Visualize 3D packing with animations
- Track volume and weight utilization
- Rotate/pan/zoom the view

---

## 📖 Quick Code Examples

### 3D Bin Packing

```python
from src.packing.box import Box, BoxType
from src.packing.container import Container, ContainerType
from src.packing.bin_packer import BinPacker, PackingStrategy

# Create a truck container
truck = Container("TRUCK-1", ContainerType.BOX_TRUCK, 
                  length=600, width=250, height=270, max_weight=3000)

# Create packages
packages = [
    Box("PKG-1", BoxType.MEDIUM, 60, 45, 35, 15, delivery_sequence=1),
    Box("PKG-2", BoxType.SMALL, 35, 30, 25, 8, delivery_sequence=2),
    Box("PKG-3", BoxType.LARGE, 80, 60, 50, 30, delivery_sequence=3),
]

# Pack with sequence awareness (LIFO)
packer = BinPacker(strategy=PackingStrategy.SEQUENCE_AWARE)
result = packer.pack(packages, truck)

print(f"Packed: {len(result.packed_boxes)}/{len(packages)}")
print(f"Volume utilization: {result.volume_utilization:.1f}%")
```

### Route Optimization

```python
from src.routing.distance import Location, DistanceMatrix
from src.routing.vrp_solver import VRPSolver

# Define locations
locations = [
    Location("DEPOT", "Warehouse", 40.7128, -74.0060),
    Location("A", "Customer A", 40.7589, -73.9851),
    Location("B", "Customer B", 40.6782, -73.9442),
    Location("C", "Customer C", 40.7282, -73.7949),
]

# Create distance matrix
matrix = DistanceMatrix.from_locations(locations)

# Solve VRP
solver = VRPSolver(matrix)
solution = solver.solve(
    depot_id="DEPOT",
    demands={"DEPOT": 0, "A": 3, "B": 5, "C": 2},
    vehicle_capacity=10,
    num_vehicles=1
)

print(f"Route: {' → '.join(solution.routes[0].stops)}")
print(f"Distance: {solution.total_distance:.2f} km")
```

### Complete Planning

```python
from src.planning.delivery_planner import DeliveryPlanner, DeliveryOrder
from src.vehicles.vehicle import Vehicle, VehicleType
from src.vehicles.fleet import Fleet
from src.resources.driver import Driver, DriverPool, DriverSkill

# Setup fleet
fleet = Fleet()
fleet.add_vehicle(Vehicle("V1", VehicleType.BOX_TRUCK, 3000, 40_000_000))

# Setup drivers
pool = DriverPool()
pool.add_driver(Driver("D1", "John", [DriverSkill.STANDARD]))

# Create planner
depot = Location("DEPOT", "Warehouse", 40.7128, -74.0060)
planner = DeliveryPlanner(depot, fleet, pool)

# Plan deliveries
orders = [...]  # Your delivery orders
plan = planner.plan_deliveries(orders)

print(f"Orders assigned: {len(plan.assigned_orders)}")
print(f"Total distance: {plan.total_distance:.2f} km")
```

---

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bin_packing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
delivery-planning-system/
├── src/
│   ├── packing/          # 3D bin packing
│   ├── routing/          # Route optimization
│   ├── resources/        # Driver/manpower planning
│   ├── vehicles/         # Fleet management
│   ├── planning/         # Delivery planning
│   └── utils/            # Visualization & metrics
├── tests/                # Test suite
├── docs/                 # Documentation & visualization
├── data/                 # Sample data
├── demo.py               # Interactive demo
└── requirements.txt      # Dependencies
```

---

## 🔗 Next Steps

1. **Load sample data**: Check `data/sample_orders.json`
2. **Customize config**: Edit `config/config.yaml`
3. **Explore the API**: See full documentation in `docs/`
4. **Build the API**: Implement REST endpoints with FastAPI

## 📚 Documentation

- [Full README](README.md)
- [API Documentation](docs/api.md) *(coming soon)*
- [Algorithm Details](docs/algorithms.md) *(coming soon)*
