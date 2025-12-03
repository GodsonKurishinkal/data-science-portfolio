# Delivery Planning System

A comprehensive delivery planning and optimization system featuring 3D bin packing, route optimization, resource/manpower planning, and vehicle fleet management with interactive 3D visualization.

## Features

### 🎯 Core Capabilities

1. **3D Bin Packing Algorithm**
   - Advanced 3D bin packing using Extreme Points algorithm
   - Considers box dimensions (length, width, height)
   - Weight constraints and load balancing
   - Delivery sequence optimization (LIFO - Last In, First Out)
   - Fragility and stacking constraints
   - Multiple container/vehicle support

2. **Route Optimization**
   - Vehicle Routing Problem (VRP) solver
   - Time window constraints
   - Capacity constraints
   - Multi-depot support
   - Real-time traffic consideration

3. **Resource & Manpower Planning**
   - Driver scheduling and assignment
   - Skill-based allocation
   - Working hours and break management
   - Load balancer for workload distribution

4. **Vehicle Fleet Management**
   - Vehicle capacity modeling
   - Maintenance scheduling
   - Fuel consumption estimation
   - Fleet utilization optimization

5. **Interactive 3D Visualization**
   - Real-time 3D truck loading simulation
   - Step-by-step packing animation
   - Interactive rotation and zoom
   - Color-coded packages by delivery sequence

## Project Structure

```
delivery-planning-system/
├── config/
│   └── config.yaml           # System configuration
├── data/
│   └── sample_orders.json    # Sample delivery orders
├── docs/
│   ├── 3d_visualization.html # Interactive 3D packing simulation
│   └── api_reference.md      # API documentation
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_bin_packing_demo.ipynb
├── scripts/
│   └── run_optimization.py
├── src/
│   ├── __init__.py
│   ├── packing/
│   │   ├── __init__.py
│   │   ├── bin_packer.py     # 3D bin packing algorithm
│   │   ├── box.py            # Box/Item representation
│   │   └── container.py      # Container/Vehicle representation
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── vrp_solver.py     # Vehicle routing solver
│   │   └── distance.py       # Distance calculations
│   ├── resources/
│   │   ├── __init__.py
│   │   ├── driver.py         # Driver management
│   │   └── scheduler.py      # Resource scheduling
│   ├── vehicles/
│   │   ├── __init__.py
│   │   ├── fleet.py          # Fleet management
│   │   └── vehicle.py        # Vehicle models
│   ├── planning/
│   │   ├── __init__.py
│   │   └── delivery_planner.py # Main planning orchestrator
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualizer.py     # Matplotlib visualizations
│   │   └── metrics.py        # Performance metrics
│   └── api/
│       ├── __init__.py
│       └── endpoints.py      # REST API endpoints
├── tests/
│   ├── conftest.py
│   ├── test_bin_packing.py
│   ├── test_routing.py
│   └── test_planning.py
├── demo.py                   # Interactive demo
├── requirements.txt
├── setup.py
└── CLAUDE.md
```

## Installation

```bash
# Clone the repository
cd data-science-portfolio/delivery-planning-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Run the Demo

```bash
python demo.py
```

### 2. Use the 3D Visualization

Open `docs/3d_visualization.html` in a web browser to see the interactive 3D truck loading simulation.

### 3. Python API Usage

```python
from src.packing import BinPacker, Box, Container
from src.planning import DeliveryPlanner

# Create boxes to pack
boxes = [
    Box(id="PKG001", length=40, width=30, height=30, weight=10, sequence=1),
    Box(id="PKG002", length=50, width=40, height=35, weight=15, sequence=2),
    Box(id="PKG003", length=30, width=25, height=20, weight=5, sequence=3),
]

# Create a delivery truck container
truck = Container(
    id="TRUCK01",
    length=600,  # cm
    width=250,
    height=270,
    max_weight=10000  # kg
)

# Pack the boxes
packer = BinPacker()
result = packer.pack(boxes, truck, optimize_sequence=True)

# Get packing solution
print(f"Packed {len(result.packed_boxes)} boxes")
print(f"Space utilization: {result.utilization:.1%}")
```

## Algorithm Details

### 3D Bin Packing

The system uses the **Extreme Points (EP)** algorithm with enhancements:

1. **Extreme Points**: Tracks potential placement positions at box corners
2. **Best Fit Decreasing (BFD)**: Sorts items by volume for better packing
3. **Sequence Awareness**: Ensures delivery order (LIFO loading)
4. **Weight Distribution**: Balances load for vehicle stability

### Route Optimization

Uses a hybrid approach combining:
- **Nearest Neighbor** heuristic for initial solution
- **2-opt** and **Or-opt** local search improvements
- **Constraint handling** for time windows and capacity

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Plotly (for 3D visualizations)
- NetworkX (for routing)
- OR-Tools (optional, for advanced optimization)

## License

MIT License - see LICENSE file for details

## Author

Godson Kurishinkal

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.
