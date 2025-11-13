# CLAUDE.md - Project 004: Supply Chain Network Optimization & Route Planning

This file provides guidance to Claude Code when working with the **Supply Chain Network Optimization** project.

## Project Overview

A comprehensive supply chain network optimization system that solves facility location, vehicle routing, and multi-echelon inventory problems to minimize costs while maintaining service levels. This project combines **operations research**, **graph algorithms**, and **optimization techniques** for strategic and tactical logistics planning.

**Status**: 🚧 Template Ready (Architecture defined, awaiting implementation)
**Implementation Priority**: High
**Complexity**: Advanced

## Quick Start

```bash
# Navigate to project
cd data-science-portfolio/project-004-supply-chain-network-optimization

# Activate shared virtual environment
source ../activate.sh

# Install dependencies (when implementing)
pip install -r requirements.txt
pip install -e .

# Run demo (when implemented)
python demo.py

# Run tests (when implemented)
pytest tests/ -v
```

## Project Architecture

### Directory Structure

```
project-004-supply-chain-network-optimization/
├── src/
│   ├── network/
│   │   ├── facility_location.py   # DC location optimization (MILP)
│   │   ├── network_design.py      # Flow optimization
│   │   └── assignment.py          # Store-DC assignments
│   ├── routing/
│   │   ├── vrp_solver.py          # Vehicle Routing Problem
│   │   ├── tsp_solver.py          # Traveling Salesman Problem
│   │   └── route_optimizer.py     # Multi-vehicle routing
│   ├── inventory/
│   │   ├── multi_echelon.py       # Multi-level inventory
│   │   └── allocation.py          # Inventory positioning
│   ├── costs/
│   │   ├── calculator.py          # Total cost modeling
│   │   └── tradeoffs.py           # Cost-service analysis
│   └── utils/
│       ├── distance.py            # Distance calculations
│       ├── graph_utils.py         # Network graphs
│       └── visualizers.py         # Map visualizations
├── notebooks/
│   ├── 01_network_analysis.ipynb
│   ├── 02_facility_location.ipynb
│   ├── 03_vehicle_routing.ipynb
│   └── 04_multi_echelon_inventory.ipynb
├── data/
│   ├── stores.csv                 # Store locations (lat/lon)
│   ├── dc_candidates.csv          # Potential DC sites
│   └── distance_matrix.csv        # Travel distances/times
├── models/                        # Optimization model outputs
├── config/
│   └── config.yaml
├── tests/
├── demo.py
└── README.md
```

### Core Problem Types

#### 1. Facility Location Problem (FLP)

**Goal**: Determine which distribution centers (DCs) to open and how to assign stores to them.

**Formulation** (Capacitated Facility Location Problem):

**Decision Variables**:
- y_j ∈ {0, 1}: 1 if DC j is open, 0 otherwise
- x_ij ∈ {0, 1}: 1 if store i is served by DC j

**Objective**:
$$\text{Minimize: } \sum_{j} f_j y_j + \sum_{i,j} c_{ij} d_i x_{ij}$$

Where:
- f_j = Fixed cost to open DC j
- c_ij = Transportation cost per unit from DC j to store i
- d_i = Demand at store i

**Constraints**:
1. Each store assigned to exactly one DC:
   $$\sum_{j} x_{ij} = 1 \quad \forall i$$

2. Capacity limits at each DC:
   $$\sum_{i} d_i x_{ij} \leq Q_j y_j \quad \forall j$$

3. Binary variables:
   $$x_{ij}, y_j \in \{0, 1\}$$

**Example**:
```
Scenario: 10 stores, 5 potential DC locations
Objective: Minimize fixed DC costs + transportation costs
Result: Open 3 DCs → $8.8M annual cost (vs. $10.2M with all 5)
```

#### 2. Vehicle Routing Problem (VRP)

**Goal**: Design delivery routes that minimize travel distance/time while meeting constraints.

**Formulation** (Capacitated VRP with Time Windows):

**Given**:
- Set of customers with demands d_i
- Vehicle capacity Q
- Time windows [e_i, l_i] for each customer
- Distance matrix c_ij

**Objective**:
$$\text{Minimize: } \sum_{k} \sum_{(i,j) \in \text{Routes}_k} c_{ij}$$

**Constraints**:
1. Each customer visited exactly once
2. Vehicle capacity not exceeded
3. Service within time windows
4. Routes start and end at depot

**Variants**:
- **Basic VRP**: Just capacity constraints
- **VRPTW**: With time windows
- **VRPPD**: With pickups and deliveries
- **MDVRP**: Multiple depots
- **SDVRP**: Split deliveries allowed

#### 3. Multi-Echelon Inventory

**Goal**: Optimize inventory placement across supply chain levels.

**Structure**:
```
National DC (Echelon 1)
    ├── Regional DC 1 (Echelon 2)
    │   ├── Store 1 (Echelon 3)
    │   └── Store 2
    └── Regional DC 2
        ├── Store 3
        └── Store 4
```

**Trade-offs**:
- **Centralization**: Lower inventory (risk pooling) but slower response
- **Decentralization**: Faster response but higher inventory

**Optimization**: Minimize total inventory while meeting service level

### Key Modules

#### 1. Facility Location Optimizer (`src/network/facility_location.py`)

**Purpose**: Solve the facility location problem using MILP.

**Key Classes**:

**`FacilityLocationOptimizer`** - MILP solver for DC location
- `optimize(stores, dc_candidates, demand)` - Main optimization
- `evaluate_scenario(open_dcs)` - Evaluate specific DC configuration
- `sensitivity_analysis(parameter, range)` - Analyze parameter impact

**Required Libraries**:
- **PuLP**: Linear programming modeling
- **OR-Tools**: Google's optimization library (alternative)
- **Gurobi/CPLEX**: Commercial solvers (optional, best performance)

**Implementation Pattern**:
```python
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

class FacilityLocationOptimizer:
    def __init__(self, fixed_costs, capacities, transport_cost_per_mile):
        self.fixed_costs = fixed_costs
        self.capacities = capacities
        self.transport_cost = transport_cost_per_mile

    def optimize(self, stores, demand, max_dcs=None):
        # Create MILP model
        model = LpProblem("FacilityLocation", LpMinimize)

        # Decision variables
        y = {}  # DC open/close
        x = {}  # Store-DC assignments

        for j in dc_candidates:
            y[j] = LpVariable(f"DC_{j}", cat='Binary')

        for i in stores:
            for j in dc_candidates:
                x[i,j] = LpVariable(f"Store_{i}_DC_{j}", cat='Binary')

        # Objective: Minimize total cost
        model += (
            lpSum([self.fixed_costs[j] * y[j] for j in dc_candidates]) +
            lpSum([self.transport_cost * distance[i,j] * demand[i] * x[i,j]
                   for i in stores for j in dc_candidates])
        )

        # Constraint 1: Each store assigned to one DC
        for i in stores:
            model += lpSum([x[i,j] for j in dc_candidates]) == 1

        # Constraint 2: Capacity limits
        for j in dc_candidates:
            model += lpSum([demand[i] * x[i,j] for i in stores]) <= \
                     self.capacities[j] * y[j]

        # Constraint 3: Max DCs (optional)
        if max_dcs:
            model += lpSum([y[j] for j in dc_candidates]) <= max_dcs

        # Solve
        model.solve()

        # Extract solution
        open_dcs = [j for j in dc_candidates if y[j].value() == 1]
        assignments = {i: j for i in stores for j in dc_candidates
                       if x[i,j].value() == 1}

        return {
            'open_dcs': open_dcs,
            'assignments': assignments,
            'total_cost': model.objective.value(),
            'fixed_cost': sum(self.fixed_costs[j] for j in open_dcs),
            'transport_cost': model.objective.value() - \
                              sum(self.fixed_costs[j] for j in open_dcs)
        }
```

**Usage**:
```python
from src.network import FacilityLocationOptimizer

optimizer = FacilityLocationOptimizer(
    fixed_costs={'DC1': 500000, 'DC2': 450000, 'DC3': 600000, ...},
    capacities={'DC1': 10000, 'DC2': 8000, 'DC3': 12000, ...},
    transport_cost_per_mile=0.50
)

solution = optimizer.optimize(
    stores=store_locations,
    demand=demand_data,
    max_dcs=5
)

print(f"Open DCs: {solution['open_dcs']}")
print(f"Total Cost: ${solution['total_cost']:,.0f}")
print(f"Fixed Cost: ${solution['fixed_cost']:,.0f}")
print(f"Transport Cost: ${solution['transport_cost']:,.0f}")
```

#### 2. Vehicle Routing Solver (`src/routing/vrp_solver.py`)

**Purpose**: Solve Vehicle Routing Problems with various constraints.

**Key Classes**:

**`VRPSolver`** - Multi-method VRP solver
- `solve_basic(depot, customers, demands)` - Basic VRP
- `solve_with_time_windows(depot, customers, time_windows)` - VRPTW
- `solve_with_priorities(depot, customers, priorities)` - Prioritized routing

**Solving Methods**:

1. **Google OR-Tools** (recommended):
```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class VRPSolver:
    def solve_with_ortools(self, depot, customers, demands, capacity):
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(customers) + 1,  # +1 for depot
            num_vehicles,
            depot
        )
        routing = pywrapcp.RoutingModel(manager)

        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [capacity] * num_vehicles,
            True,  # start cumul to zero
            'Capacity'
        )

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        # Extract routes
        routes = []
        for vehicle_id in range(num_vehicles):
            route = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            routes.append(route)

        return routes, solution.ObjectiveValue()
```

2. **Heuristics** (for quick approximations):
```python
def nearest_neighbor_heuristic(depot, customers, capacity):
    """Fast but suboptimal."""
    routes = []
    unvisited = set(customers)

    while unvisited:
        route = [depot]
        current = depot
        load = 0

        while unvisited:
            # Find nearest unvisited customer
            nearest = min(unvisited,
                         key=lambda c: distance(current, c))

            if load + demand[nearest] <= capacity:
                route.append(nearest)
                current = nearest
                load += demand[nearest]
                unvisited.remove(nearest)
            else:
                break

        route.append(depot)
        routes.append(route)

    return routes
```

3. **Genetic Algorithm** (for complex scenarios):
```python
def genetic_algorithm_vrp(depot, customers, population_size=100, generations=1000):
    """Evolutionary approach for complex VRP."""
    # Initialize population of random solutions
    population = [generate_random_solution() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness (total distance)
        fitness = [evaluate_solution(sol) for sol in population]

        # Selection
        parents = tournament_selection(population, fitness)

        # Crossover
        offspring = crossover(parents)

        # Mutation
        offspring = mutate(offspring)

        # Replace worst solutions
        population = elitist_replacement(population, offspring, fitness)

    return best_solution(population)
```

**Usage**:
```python
from src.routing import VRPSolver

solver = VRPSolver(
    num_vehicles=3,
    vehicle_capacity=1000,
    max_route_duration=480  # 8 hours in minutes
)

routes = solver.solve_basic(
    depot='DC1',
    customers=['Store1', 'Store2', ..., 'Store10'],
    demands={'Store1': 150, 'Store2': 200, ...}
)

print(f"Number of routes: {len(routes)}")
for i, route in enumerate(routes):
    print(f"\nRoute {i+1}: {' -> '.join(route)}")
    print(f"Distance: {calculate_route_distance(route):.1f} miles")
    print(f"Load: {sum(demands[c] for c in route if c != depot)} units")
```

#### 3. Multi-Echelon Inventory Optimizer (`src/inventory/multi_echelon.py`)

**Purpose**: Optimize inventory allocation across supply chain levels.

**Key Concepts**:

**Risk Pooling**: Centralizing inventory reduces total safety stock needed.

**Formula**:
$$SS_{\text{centralized}} = Z \times \sigma_{\text{pooled}} \times \sqrt{LT}$$

Where:
$$\sigma_{\text{pooled}} = \sqrt{\sum_{i=1}^{n} \sigma_i^2}$$

For independent demands, pooled std dev < sum of individual std devs.

**Example**:
```
Individual stores: SS = 10 + 10 + 10 = 30 units
Pooled at DC: SS = √(10² + 10² + 10²) = 17.3 units
Savings: 42% reduction in safety stock!
```

**Implementation**:
```python
class MultiEchelonOptimizer:
    def __init__(self, echelons, service_level=0.95):
        self.echelons = echelons  # ['National', 'Regional', 'Store']
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)

    def optimize_safety_stock(self, demand_data, lead_times):
        """
        Allocate safety stock across echelons.

        Args:
            demand_data: Dict of demand statistics by location
            lead_times: Dict of lead times by echelon

        Returns:
            Optimal safety stock allocation
        """
        allocation = {}

        # Calculate pooled demand at each echelon
        for echelon in self.echelons:
            locations = self.get_locations_in_echelon(echelon)

            # Pool demand variance
            pooled_variance = sum(
                demand_data[loc]['variance'] for loc in locations
            )
            pooled_std = np.sqrt(pooled_variance)

            # Calculate safety stock
            lt = lead_times[echelon]
            ss = self.z_score * pooled_std * np.sqrt(lt)

            allocation[echelon] = {
                'safety_stock': ss,
                'locations': locations,
                'pooled_std': pooled_std,
                'lead_time': lt
            }

        return allocation

    def calculate_inventory_position(self, allocation, on_hand, in_transit):
        """Calculate total inventory position."""
        ip = {}
        for echelon, data in allocation.items():
            ip[echelon] = {
                'safety_stock': data['safety_stock'],
                'on_hand': on_hand[echelon],
                'in_transit': in_transit[echelon],
                'inventory_position': on_hand[echelon] + in_transit[echelon],
                'review_point': data['safety_stock'] + \
                                self.calculate_expected_demand(echelon)
            }
        return ip
```

**Usage**:
```python
from src.inventory import MultiEchelonOptimizer

optimizer = MultiEchelonOptimizer(
    echelons=['National', 'Regional', 'Store'],
    service_level=0.95
)

allocation = optimizer.optimize_safety_stock(
    demand_data=demand_stats,
    lead_times={'National': 14, 'Regional': 7, 'Store': 0}
)

print("\nSafety Stock Allocation:")
for echelon, data in allocation.items():
    print(f"{echelon}: {data['safety_stock']:.0f} units "
          f"(LT: {data['lead_time']} days)")

# Calculate total inventory savings
total_independent = sum(calculate_independent_ss(loc)
                       for loc in all_locations)
total_pooled = sum(data['safety_stock']
                   for data in allocation.values())
savings_pct = (1 - total_pooled / total_independent) * 100

print(f"\nInventory Savings: {savings_pct:.1f}%")
```

#### 4. Distance Calculator (`src/utils/distance.py`)

**Purpose**: Calculate distances and travel times between locations.

**Methods**:

1. **Haversine Distance** (great-circle distance):
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points."""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c
```

2. **Road Distance** (using routing APIs):
```python
def get_road_distance(origin, destination, mode='driving'):
    """Get actual road distance using Google Maps API."""
    import googlemaps

    gmaps = googlemaps.Client(key=API_KEY)
    result = gmaps.distance_matrix(origin, destination, mode=mode)

    distance_m = result['rows'][0]['elements'][0]['distance']['value']
    duration_s = result['rows'][0]['elements'][0]['duration']['value']

    return distance_m / 1000, duration_s / 60  # km, minutes
```

3. **Distance Matrix** (precompute for efficiency):
```python
def build_distance_matrix(locations):
    """Precompute all pairwise distances."""
    n = len(locations)
    distance_matrix = np.zeros((n, n))

    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i != j:
                distance_matrix[i, j] = haversine_distance(
                    loc1['lat'], loc1['lon'],
                    loc2['lat'], loc2['lon']
                )

    return distance_matrix
```

#### 5. Visualization (`src/utils/visualizers.py`)

**Purpose**: Create interactive maps and network visualizations.

**Using Folium** (interactive maps):
```python
import folium

def visualize_network(stores, dcs, assignments):
    """Create interactive map of DC-store network."""
    # Center map on mean location
    center_lat = np.mean([s['lat'] for s in stores])
    center_lon = np.mean([s['lon'] for s in stores])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    # Add DCs
    for dc in dcs:
        folium.Marker(
            location=[dc['lat'], dc['lon']],
            popup=f"DC: {dc['name']}",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)

    # Add stores and connections
    for store in stores:
        assigned_dc = assignments[store['id']]

        # Store marker
        folium.CircleMarker(
            location=[store['lat'], store['lon']],
            radius=5,
            popup=f"Store: {store['name']}",
            color='blue'
        ).add_to(m)

        # Connection line
        folium.PolyLine(
            locations=[
                [store['lat'], store['lon']],
                [assigned_dc['lat'], assigned_dc['lon']]
            ],
            color='gray',
            weight=1,
            opacity=0.5
        ).add_to(m)

    return m
```

**Using NetworkX** (graph analysis):
```python
import networkx as nx

def create_network_graph(stores, dcs, assignments):
    """Create network graph for analysis."""
    G = nx.Graph()

    # Add DC nodes
    for dc in dcs:
        G.add_node(dc['id'], type='dc', capacity=dc['capacity'])

    # Add store nodes and edges
    for store in stores:
        G.add_node(store['id'], type='store', demand=store['demand'])
        assigned_dc = assignments[store['id']]
        G.add_edge(store['id'], assigned_dc,
                   distance=calculate_distance(store, assigned_dc))

    return G

# Analysis
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)
```

## Configuration

### config/config.yaml

```yaml
data:
  stores_file: "data/stores.csv"
  dc_candidates_file: "data/dc_candidates.csv"
  distance_matrix_file: "data/distance_matrix.csv"

facility_location:
  max_dcs: 10
  min_dcs: 3
  transport_cost_per_mile: 0.50  # $/mile
  fixed_costs:
    large_dc: 600000    # $/year
    medium_dc: 450000
    small_dc: 300000
  capacities:
    large_dc: 12000     # units/day
    medium_dc: 8000
    small_dc: 5000

vehicle_routing:
  num_vehicles: 5
  vehicle_capacity: 1000  # units
  max_route_duration: 480  # minutes (8 hours)
  speed_mph: 45
  cost_per_mile: 0.85

multi_echelon:
  echelons:
    - name: "National"
      lead_time: 14
      service_level: 0.95
    - name: "Regional"
      lead_time: 7
      service_level: 0.98
    - name: "Store"
      lead_time: 0
      service_level: 0.99

costs:
  holding_cost_rate: 0.25  # 25% of product value per year
  stockout_penalty: 5.0    # Multiple of unit cost
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure
- [ ] Implement distance calculations
- [ ] Create data loaders for stores/DCs
- [ ] Build visualization utilities
- [ ] Write basic tests

### Phase 2: Facility Location (Week 3-4)
- [ ] Implement MILP solver with PuLP
- [ ] Add capacity constraints
- [ ] Implement scenario analysis
- [ ] Create sensitivity analysis tools
- [ ] Build interactive visualizations
- [ ] Test on M5 store locations

### Phase 3: Vehicle Routing (Week 5-6)
- [ ] Implement OR-Tools VRP solver
- [ ] Add time window constraints
- [ ] Implement heuristic solvers (backup)
- [ ] Create route visualization
- [ ] Calculate route costs
- [ ] Test with varying fleet sizes

### Phase 4: Multi-Echelon Inventory (Week 7-8)
- [ ] Implement risk pooling calculations
- [ ] Build multi-level optimization
- [ ] Create inventory positioning logic
- [ ] Calculate cost savings
- [ ] Visualize inventory flows
- [ ] Integrate with Projects 1-2

### Phase 5: Integration & Testing (Week 9-10)
- [ ] End-to-end optimization pipeline
- [ ] Comprehensive testing suite
- [ ] Performance benchmarking
- [ ] Documentation and notebooks
- [ ] Demo and visualization polish

## Required Libraries

```txt
# Optimization
pulp>=2.7.0
ortools>=9.5.0
scipy>=1.9.0
cvxpy>=1.3.0

# Network & Graph Analysis
networkx>=3.0
graph-tool>=2.45  # Optional, better performance

# Geospatial
geopy>=2.3.0
shapely>=2.0.0
folium>=0.14.0

# Routing APIs (optional)
googlemaps>=4.10.0

# Visualization
plotly>=5.13.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Key Algorithms & Techniques

### 1. Mixed Integer Linear Programming (MILP)

**Applications**:
- Facility location
- Production planning
- Network design

**Solvers**:
- **PuLP**: Open-source, Python interface to CBC/GLPK
- **OR-Tools**: Google's solver, very fast
- **Gurobi/CPLEX**: Commercial, best performance

**When to Use**: Exact solutions needed, problem size < 100,000 variables

### 2. Constraint Programming (CP)

**Applications**:
- Scheduling
- Vehicle routing with complex constraints
- Resource allocation

**Tools**: OR-Tools CP-SAT Solver

**When to Use**: Many logical constraints, not purely numerical

### 3. Metaheuristics

**Algorithms**:
- Genetic Algorithm
- Simulated Annealing
- Tabu Search
- Ant Colony Optimization

**When to Use**: Large-scale problems, good solution needed quickly

### 4. Graph Algorithms

**Algorithms**:
- Dijkstra's Algorithm: Shortest path
- Bellman-Ford: Shortest path with negative weights
- Floyd-Warshall: All-pairs shortest path
- Minimum Spanning Tree: Network connectivity

**When to Use**: Network analysis, routing, connectivity

## Testing Strategy

### Unit Tests
- Distance calculations
- Cost functions
- Graph construction
- Individual optimizers

### Integration Tests
- Full optimization pipeline
- Data loading → optimization → visualization
- Multi-module workflows

### Performance Tests
- Large-scale problem instances
- Optimization time limits
- Memory usage

### Validation Tests
- Compare to known optimal solutions
- Benchmark against literature
- Sanity checks on costs/distances

## Common Pitfalls & Solutions

### 1. Infeasible MILP Models

**Problem**: Solver returns "infeasible"

**Debug**:
```python
# Relax constraints one by one
# Add slack variables
for j in dc_candidates:
    slack = LpVariable(f"slack_{j}", lowBound=0)
    model += lpSum([demand[i] * x[i,j] for i in stores]) <= \
             capacities[j] * y[j] + slack
```

### 2. Slow VRP Solving

**Solution**: Use time limits and heuristics
```python
search_parameters.time_limit.seconds = 30
search_parameters.first_solution_strategy = \
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
```

### 3. Distance Matrix Errors

**Check**:
- Symmetric (distance i→j = j→i)
- Triangle inequality (d_ij ≤ d_ik + d_kj)
- No negative distances
- Diagonal is zero

### 4. Scaling Issues

**Solution**: Normalize units
```python
# Convert to consistent units
distances_km = distances_m / 1000
costs_thousands = costs / 1000
```

## Expected Results

### Facility Location
- **DC Count Reduction**: 12 → 8 DCs (-33%)
- **Cost Savings**: $1.4M annually (-14%)
- **Service Level**: Maintained at 98%+
- **Solving Time**: < 60 seconds for 100 stores

### Vehicle Routing
- **Distance Reduction**: 17-20% vs. naive routing
- **Vehicle Utilization**: 85-90%
- **On-Time Delivery**: 95%+
- **Solving Time**: < 5 minutes for 50 stops

### Multi-Echelon Inventory
- **Inventory Reduction**: 20-25% via risk pooling
- **Service Level**: Maintained at 95%+
- **Cost Savings**: $1.8M in holding costs

## Additional Resources

- **[README.md](README.md)** - Project overview
- **INFORMS**: Operations research society (papers, case studies)
- **OR-Tools Documentation**: https://developers.google.com/optimization
- **PuLP Documentation**: https://coin-or.github.io/pulp/

## References

- Chopra, S., & Meindl, P. (2016). *Supply Chain Management*
- Daskin, M. S. (2013). *Network and Discrete Location*
- Toth, P., & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications*
- Simchi-Levi, D., et al. (2008). *Designing and Managing the Supply Chain*

## Contact

**Godson Kurishinkal**
- GitHub: [@GodsonKurishinkal](https://github.com/GodsonKurishinkal)
- LinkedIn: [linkedin.com/in/godsonkurishinkal](https://www.linkedin.com/in/godsonkurishinkal)
- Email: godson.kurishinkal+github@gmail.com
