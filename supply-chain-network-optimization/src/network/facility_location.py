"""Facility Location Optimization using Mixed Integer Programming."""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacilityLocationOptimizer:
    """
    Solve the Capacitated Facility Location Problem (CFLP).

    Objective: Minimize total cost = fixed costs + transportation costs
    Constraints:
        - All stores must be served
        - Facility capacity constraints
        - Binary facility open/close decisions
    """

    def __init__(self, fixed_costs: Dict[str, float],
                 capacities: Dict[str, float],
                 transportation_cost_per_mile: float = 0.50):
        """
        Initialize facility location optimizer.

        Parameters
        ----------
        fixed_costs : Dict[str, float]
            Fixed annual cost for each potential facility
        capacities : Dict[str, float]
            Capacity limit for each facility (in units)
        transportation_cost_per_mile : float
            Transportation cost per mile
        """
        self.fixed_costs = fixed_costs
        self.capacities = capacities
        self.transport_cost = transportation_cost_per_mile
        self.model = None
        self.solution = None

    def optimize(self, stores: pd.DataFrame, demand: pd.Series,
                distance_matrix: pd.DataFrame,
                max_facilities: Optional[int] = None,
                min_facilities: Optional[int] = None,
                single_sourcing: bool = True,
                time_limit: int = 300) -> Dict:
        """
        Solve facility location problem.

        Parameters
        ----------
        stores : pd.DataFrame
            Store information with 'id' column
        demand : pd.Series
            Demand for each store (index = store_id)
        distance_matrix : pd.DataFrame
            Distance from each facility to each store
        max_facilities : Optional[int]
            Maximum number of facilities to open
        min_facilities : Optional[int]
            Minimum number of facilities to open
        single_sourcing : bool
            If True, each store served by only one facility
        time_limit : int
            Solver time limit in seconds

        Returns
        -------
        Dict
            Solution with open facilities, assignments, and costs
        """
        logger.info("Building facility location optimization model...")

        # Create problem
        self.model = LpProblem("Facility_Location", LpMinimize)

        # Sets
        facilities = list(self.fixed_costs.keys())
        store_ids = stores['id'].tolist()

        # Decision variables
        # y[f] = 1 if facility f is open, 0 otherwise
        y = LpVariable.dicts("facility_open", facilities, cat='Binary')

        # x[f,s] = fraction of demand from store s served by facility f
        x = LpVariable.dicts("assignment",
                            [(f, s) for f in facilities for s in store_ids],
                            lowBound=0, upBound=1, cat='Continuous')

        # Objective function
        fixed_cost = lpSum([self.fixed_costs[f] * y[f] for f in facilities])

        transport_cost = lpSum([
            distance_matrix.loc[f, s] * self.transport_cost * demand[s] * x[(f, s)]
            for f in facilities for s in store_ids
            if f in distance_matrix.index and s in distance_matrix.columns
        ])

        self.model += fixed_cost + transport_cost, "Total_Cost"

        # Constraints
        # 1. Each store's demand must be satisfied
        for s in store_ids:
            self.model += (
                lpSum([x[(f, s)] for f in facilities]) == 1,
                f"Demand_Satisfaction_{s}"
            )

        # 2. Capacity constraints
        for f in facilities:
            self.model += (
                lpSum([demand[s] * x[(f, s)] for s in store_ids]) <=
                self.capacities[f] * y[f],
                f"Capacity_{f}"
            )

        # 3. Can only assign to open facilities
        for f in facilities:
            for s in store_ids:
                self.model += (
                    x[(f, s)] <= y[f],
                    f"Open_Facility_{f}_{s}"
                )

        # 4. Maximum number of facilities
        if max_facilities is not None:
            self.model += (
                lpSum([y[f] for f in facilities]) <= max_facilities,
                "Max_Facilities"
            )

        # 5. Minimum number of facilities
        if min_facilities is not None:
            self.model += (
                lpSum([y[f] for f in facilities]) >= min_facilities,
                "Min_Facilities"
            )

        # 6. Single sourcing (optional)
        if single_sourcing:
            for f in facilities:
                for s in store_ids:
                    # If x > 0, it must be 1 (make it binary)
                    x[(f, s)].cat = 'Binary'

        # Solve
        logger.info("Solving optimization problem...")
        solver = PULP_CBC_CMD(timeLimit=time_limit, msg=1)
        self.model.solve(solver)

        # Extract solution
        status = LpStatus[self.model.status]
        logger.info(f"Optimization status: {status}")

        if status == 'Optimal' or status == 'Feasible':
            self.solution = self._extract_solution(y, x, facilities, store_ids, demand)
            logger.info(f"Solution found: {len(self.solution['open_facilities'])} facilities open")
            logger.info(f"Total cost: ${self.solution['total_cost']:,.2f}")
            return self.solution
        else:
            logger.error("No solution found!")
            return {'status': status, 'error': 'No feasible solution'}

    def _extract_solution(self, y: Dict, x: Dict, facilities: List[str],
                         stores: List[str], demand: pd.Series) -> Dict:
        """Extract solution from solved model."""
        # Open facilities
        open_facilities = [f for f in facilities if y[f].varValue > 0.5]

        # Assignments
        assignments = {}
        for f in open_facilities:
            assignments[f] = []
            for s in stores:
                if x[(f, s)].varValue > 0.1:  # Small tolerance
                    assignments[f].append({
                        'store_id': s,
                        'fraction': x[(f, s)].varValue,
                        'demand': demand[s]
                    })

        # Costs
        fixed_cost = sum(self.fixed_costs[f] for f in open_facilities)
        transport_cost = value(self.model.objective) - fixed_cost

        # Utilization
        utilization = {}
        for f in open_facilities:
            total_demand = sum(
                demand[s] * x[(f, s)].varValue
                for s in stores
            )
            utilization[f] = total_demand / self.capacities[f]

        return {
            'status': 'Optimal',
            'open_facilities': open_facilities,
            'assignments': assignments,
            'num_facilities': len(open_facilities),
            'total_cost': value(self.model.objective),
            'fixed_cost': fixed_cost,
            'transport_cost': transport_cost,
            'utilization': utilization
        }

    def sensitivity_analysis(self, stores: pd.DataFrame, demand: pd.Series,
                            distance_matrix: pd.DataFrame,
                            facility_range: range = range(3, 11)) -> List[Dict]:
        """
        Run optimization for different numbers of facilities.

        Parameters
        ----------
        stores : pd.DataFrame
            Store data
        demand : pd.Series
            Demand data
        distance_matrix : pd.DataFrame
            Distance matrix
        facility_range : range
            Range of facility counts to test

        Returns
        -------
        List[Dict]
            Results for each facility count
        """
        results = []

        for num_fac in facility_range:
            logger.info(f"\nTesting with max {num_fac} facilities...")
            solution = self.optimize(
                stores=stores,
                demand=demand,
                distance_matrix=distance_matrix,
                max_facilities=num_fac,
                time_limit=120
            )

            if solution.get('status') in ['Optimal', 'Feasible']:
                results.append({
                    'num_facilities': solution['num_facilities'],
                    'total_cost': solution['total_cost'],
                    'fixed_cost': solution['fixed_cost'],
                    'transport_cost': solution['transport_cost'],
                    'avg_utilization': np.mean(list(solution['utilization'].values()))
                })

        return results

    def get_assignment_matrix(self) -> pd.DataFrame:
        """
        Get store-facility assignment matrix from solution.

        Returns
        -------
        pd.DataFrame
            Binary matrix (rows=facilities, columns=stores)
        """
        if self.solution is None:
            raise ValueError("No solution available. Run optimize() first.")

        assignments = []
        for facility, stores in self.solution['assignments'].items():
            for store_data in stores:
                assignments.append({
                    'facility_id': facility,
                    'store_id': store_data['store_id'],
                    'fraction': store_data['fraction']
                })

        df = pd.DataFrame(assignments)
        matrix = df.pivot(index='facility_id',
                         columns='store_id',
                         values='fraction').fillna(0)

        return matrix

    def export_solution(self, filepath: str):
        """Export solution to JSON file."""
        if self.solution is None:
            raise ValueError("No solution available. Run optimize() first.")

        import json
        with open(filepath, 'w') as f:
            # Convert numpy types to native Python types
            solution_copy = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.solution.items()
            }
            json.dump(solution_copy, f, indent=2)

        logger.info(f"Solution exported to {filepath}")
