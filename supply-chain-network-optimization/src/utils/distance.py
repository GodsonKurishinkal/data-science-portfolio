"""Distance calculation utilities for supply chain network optimization."""

import numpy as np
import pandas as pd
from typing import Tuple, List
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                       unit: str = 'miles') -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees
    lat2, lon2 : float
        Latitude and longitude of second point in degrees
    unit : str
        'miles' or 'km' for the distance unit

    Returns
    -------
    float
        Distance between the two points
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of Earth
    r = 3956 if unit == 'miles' else 6371  # miles or km

    return c * r


class DistanceCalculator:
    """Calculate and cache distances between locations."""

    def __init__(self, locations: pd.DataFrame, method: str = 'haversine'):
        """
        Initialize distance calculator.

        Parameters
        ----------
        locations : pd.DataFrame
            DataFrame with columns: id, latitude, longitude
        method : str
            'haversine' for great circle distance or 'euclidean'
        """
        self.locations = locations.set_index('id')
        self.method = method
        self._distance_matrix = None

    def calculate_distance(self, loc1_id: str, loc2_id: str) -> float:
        """Calculate distance between two location IDs."""
        loc1 = self.locations.loc[loc1_id]
        loc2 = self.locations.loc[loc2_id]

        if self.method == 'haversine':
            return haversine_distance(
                loc1['latitude'], loc1['longitude'],
                loc2['latitude'], loc2['longitude']
            )
        elif self.method == 'euclidean':
            return np.sqrt(
                (loc1['latitude'] - loc2['latitude'])**2 +
                (loc1['longitude'] - loc2['longitude'])**2
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def build_distance_matrix(self) -> pd.DataFrame:
        """Build and cache full distance matrix."""
        if self._distance_matrix is not None:
            return self._distance_matrix

        loc_ids = self.locations.index.tolist()
        n = len(loc_ids)
        matrix = np.zeros((n, n))

        for i, loc1 in enumerate(loc_ids):
            for j, loc2 in enumerate(loc_ids):
                if i < j:  # Only calculate upper triangle
                    dist = self.calculate_distance(loc1, loc2)
                    matrix[i, j] = dist
                    matrix[j, i] = dist  # Symmetric

        self._distance_matrix = pd.DataFrame(
            matrix,
            index=loc_ids,
            columns=loc_ids
        )
        return self._distance_matrix

    def get_distance_matrix(self) -> pd.DataFrame:
        """Get cached distance matrix, building if necessary."""
        if self._distance_matrix is None:
            return self.build_distance_matrix()
        return self._distance_matrix

    def nearest_neighbors(self, loc_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find n nearest neighbors to a location.

        Returns
        -------
        List of (location_id, distance) tuples sorted by distance
        """
        distances = self.get_distance_matrix()
        loc_distances = distances.loc[loc_id].sort_values()

        # Exclude self (distance = 0)
        neighbors = [(idx, dist) for idx, dist in loc_distances.items() if idx != loc_id]

        return neighbors[:n]

    def distance_to_multiple(self, from_id: str, to_ids: List[str]) -> pd.Series:
        """Calculate distances from one location to multiple locations."""
        distances = self.get_distance_matrix()
        return distances.loc[from_id, to_ids]

    def total_distance(self, route: List[str]) -> float:
        """
        Calculate total distance for a route (sequence of location IDs).

        Parameters
        ----------
        route : List[str]
            Ordered list of location IDs

        Returns
        -------
        float
            Total distance of the route
        """
        if len(route) < 2:
            return 0.0

        total = 0.0
        for i in range(len(route) - 1):
            total += self.calculate_distance(route[i], route[i+1])

        return total

    def cost_matrix(self, cost_per_unit: float = 0.50) -> pd.DataFrame:
        """
        Convert distance matrix to cost matrix.

        Parameters
        ----------
        cost_per_unit : float
            Cost per mile (or km)

        Returns
        -------
        pd.DataFrame
            Cost matrix
        """
        distances = self.get_distance_matrix()
        return distances * cost_per_unit
