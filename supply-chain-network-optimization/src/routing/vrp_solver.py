"""Vehicle Routing Problem (VRP) solver using OR-Tools."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VRPSolver:
    """
    Solve Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
    """

    def __init__(self, num_vehicles: int, vehicle_capacity: int,
                 max_route_duration: int = 480):
        """
        Initialize VRP solver.

        Parameters
        ----------
        num_vehicles : int
            Number of vehicles available
        vehicle_capacity : int
            Capacity of each vehicle (units)
        max_route_duration : int
            Maximum route duration in minutes
        """
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.max_route_duration = max_route_duration
        self.solution = None

    def solve(self, depot: Dict, deliveries: pd.DataFrame,
             distance_matrix: np.ndarray,
             time_windows: Optional[pd.DataFrame] = None,
             service_time: int = 15) -> Dict:
        """
        Solve VRP.

        Parameters
        ----------
        depot : Dict
            Depot info with 'id', 'latitude', 'longitude'
        deliveries : pd.DataFrame
            Delivery orders with columns: id, demand, latitude, longitude
        distance_matrix : np.ndarray
            Distance matrix (depot + all delivery locations)
        time_windows : Optional[pd.DataFrame]
            Time windows with columns: location_id, earliest, latest (minutes)
        service_time : int
            Service time at each location (minutes)

        Returns
        -------
        Dict
            Solution with routes and metrics
        """
        logger.info(f"Solving VRP for {len(deliveries)} deliveries with {self.num_vehicles} vehicles...")

        # Prepare data
        data = self._prepare_data(depot, deliveries, distance_matrix,
                                 time_windows, service_time)

        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],
            True,  # start cumul to zero
            'Capacity'
        )

        # Time window constraint (if provided)
        if time_windows is not None:
            routing.AddDimension(
                transit_callback_index,
                30,  # allow waiting time
                self.max_route_duration,
                False,  # don't force start cumul to zero
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')

            for location_idx, time_window in enumerate(data['time_windows']):
                if location_idx == 0:  # Skip depot
                    continue
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            self.solution = self._extract_solution(
                manager, routing, solution, data, deliveries
            )
            logger.info(f"Solution found! Total distance: {self.solution['total_distance']:.1f} miles")
            return self.solution
        else:
            logger.error("No solution found!")
            return {'status': 'No solution'}

    def _prepare_data(self, depot: Dict, deliveries: pd.DataFrame,
                     distance_matrix: np.ndarray,
                     time_windows: Optional[pd.DataFrame],
                     service_time: int) -> Dict:
        """Prepare data for OR-Tools solver."""
        # Convert distance to integer (OR-Tools requirement)
        distance_matrix_int = (distance_matrix * 100).astype(int)

        data = {
            'distance_matrix': distance_matrix_int.tolist(),
            'demands': [0] + deliveries['demand'].tolist(),  # 0 for depot
            'vehicle_capacities': [self.vehicle_capacity] * self.num_vehicles,
            'num_vehicles': self.num_vehicles,
            'depot': 0
        }

        if time_windows is not None:
            data['time_windows'] = [(0, self.max_route_duration)]  # Depot
            for _, row in deliveries.iterrows():
                if row['id'] in time_windows['location_id'].values:
                    tw = time_windows[time_windows['location_id'] == row['id']].iloc[0]
                    data['time_windows'].append((int(tw['earliest']), int(tw['latest'])))
                else:
                    data['time_windows'].append((0, self.max_route_duration))

        return data

    def _extract_solution(self, manager, routing, solution,
                         data, deliveries: pd.DataFrame) -> Dict:
        """Extract solution from OR-Tools model."""
        routes = []
        total_distance = 0
        total_load = 0

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            route_stops = []

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]

                if node_index != 0:  # Not depot
                    delivery_idx = node_index - 1
                    route_stops.append({
                        'stop_number': len(route_stops) + 1,
                        'location_id': deliveries.iloc[delivery_idx]['id'],
                        'demand': deliveries.iloc[delivery_idx]['demand'],
                        'latitude': deliveries.iloc[delivery_idx]['latitude'],
                        'longitude': deliveries.iloc[delivery_idx]['longitude']
                    })

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            # Only include routes with stops
            if route_stops:
                routes.append({
                    'vehicle_id': f'Vehicle_{vehicle_id + 1}',
                    'stops': route_stops,
                    'distance': route_distance / 100.0,  # Convert back to miles
                    'load': route_load,
                    'utilization': route_load / self.vehicle_capacity
                })

                total_distance += route_distance
                total_load += route_load

        return {
            'status': 'Optimal',
            'routes': routes,
            'num_routes': len(routes),
            'total_distance': total_distance / 100.0,
            'total_load': total_load,
            'avg_utilization': total_load / (len(routes) * self.vehicle_capacity)
        }

    def get_routes_dataframe(self) -> pd.DataFrame:
        """Convert solution routes to DataFrame."""
        if self.solution is None or 'routes' not in self.solution:
            raise ValueError("No solution available.")

        all_stops = []
        for route in self.solution['routes']:
            for stop in route['stops']:
                all_stops.append({
                    'vehicle_id': route['vehicle_id'],
                    'stop_number': stop['stop_number'],
                    'location_id': stop['location_id'],
                    'demand': stop['demand'],
                    'latitude': stop['latitude'],
                    'longitude': stop['longitude']
                })

        return pd.DataFrame(all_stops)

    def export_solution(self, filepath: str):
        """Export solution to JSON."""
        if self.solution is None:
            raise ValueError("No solution available.")

        import json
        with open(filepath, 'w') as f:
            json.dump(self.solution, f, indent=2)

        logger.info(f"Solution exported to {filepath}")
