"""Distance calculation utilities for route optimization."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def euclidean_distance(
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: Coordinates of point 1
        x2, y2: Coordinates of point 2
        
    Returns:
        Euclidean distance
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def manhattan_distance(
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """
    Calculate Manhattan (city block) distance between two points.
    
    Args:
        x1, y1: Coordinates of point 1
        x2, y2: Coordinates of point 2
        
    Returns:
        Manhattan distance
    """
    return abs(x2 - x1) + abs(y2 - y1)


def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    method: str = "euclidean",
) -> float:
    """
    Calculate distance between two points using specified method.
    
    Args:
        point1: First point (x, y) or (lat, lon)
        point2: Second point (x, y) or (lat, lon)
        method: Distance calculation method ('euclidean', 'manhattan', 'haversine')
        
    Returns:
        Distance value
    """
    methods = {
        "euclidean": lambda p1, p2: euclidean_distance(p1[0], p1[1], p2[0], p2[1]),
        "manhattan": lambda p1, p2: manhattan_distance(p1[0], p1[1], p2[0], p2[1]),
        "haversine": lambda p1, p2: haversine_distance(p1[0], p1[1], p2[0], p2[1]),
    }
    
    calc_func = methods.get(method, methods["euclidean"])
    return calc_func(point1, point2)


@dataclass
class Location:
    """Represents a delivery location."""
    id: str
    name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    x: float = 0.0  # For grid-based coordinates
    y: float = 0.0
    address: str = ""
    service_time: float = 15.0  # Minutes
    time_window_start: Optional[float] = None  # Minutes from start of day
    time_window_end: Optional[float] = None
    demand: float = 0.0  # Capacity demand
    priority: int = 1
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        """Get coordinates as tuple."""
        if self.latitude != 0 or self.longitude != 0:
            return (self.latitude, self.longitude)
        return (self.x, self.y)
    
    @property
    def has_time_window(self) -> bool:
        """Check if location has a time window constraint."""
        return self.time_window_start is not None and self.time_window_end is not None


@dataclass
class DistanceMatrix:
    """
    Efficient storage and calculation of distances between locations.
    
    Stores a symmetric matrix of distances between all pairs of locations.
    Supports lazy calculation and caching.
    """
    locations: List[Location] = field(default_factory=list)
    distance_method: str = "euclidean"
    _matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _time_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    speed_kmh: float = 40.0  # Average speed for time calculation
    
    def __post_init__(self):
        """Initialize matrix if locations are provided."""
        if self.locations:
            self._calculate_matrix()
    
    @property
    def size(self) -> int:
        """Get number of locations."""
        return len(self.locations)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the distance matrix, calculating if needed."""
        if self._matrix is None:
            self._calculate_matrix()
        return self._matrix
    
    @property
    def time_matrix(self) -> np.ndarray:
        """Get travel time matrix in minutes."""
        if self._time_matrix is None:
            self._calculate_time_matrix()
        return self._time_matrix
    
    def _calculate_matrix(self) -> None:
        """Calculate the full distance matrix."""
        n = len(self.locations)
        self._matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = calculate_distance(
                    self.locations[i].coordinates,
                    self.locations[j].coordinates,
                    self.distance_method,
                )
                self._matrix[i, j] = dist
                self._matrix[j, i] = dist
    
    def _calculate_time_matrix(self) -> None:
        """Calculate travel time matrix from distance matrix."""
        if self._matrix is None:
            self._calculate_matrix()
        
        # Convert distance to time (hours) then to minutes
        self._time_matrix = (self._matrix / self.speed_kmh) * 60
    
    def add_location(self, location: Location) -> int:
        """
        Add a location to the matrix.
        
        Args:
            location: Location to add
            
        Returns:
            Index of the new location
        """
        self.locations.append(location)
        # Invalidate cached matrices
        self._matrix = None
        self._time_matrix = None
        return len(self.locations) - 1
    
    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """
        Get distance between two locations by index.
        
        Args:
            from_idx: Index of origin location
            to_idx: Index of destination location
            
        Returns:
            Distance between locations
        """
        return self.matrix[from_idx, to_idx]
    
    def get_time(self, from_idx: int, to_idx: int) -> float:
        """
        Get travel time between two locations by index.
        
        Args:
            from_idx: Index of origin location
            to_idx: Index of destination location
            
        Returns:
            Travel time in minutes
        """
        return self.time_matrix[from_idx, to_idx]
    
    def get_nearest_neighbors(
        self,
        from_idx: int,
        k: int = 5,
        exclude: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get k nearest neighbors of a location.
        
        Args:
            from_idx: Index of the origin location
            k: Number of neighbors to return
            exclude: Indices to exclude from results
            
        Returns:
            List of (index, distance) tuples sorted by distance
        """
        exclude = exclude or []
        exclude_set = set(exclude)
        
        distances = [
            (i, self.matrix[from_idx, i])
            for i in range(len(self.locations))
            if i != from_idx and i not in exclude_set
        ]
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_total_distance(self, path: List[int]) -> float:
        """
        Calculate total distance of a path through locations.
        
        Args:
            path: List of location indices
            
        Returns:
            Total path distance
        """
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            total += self.matrix[path[i], path[i + 1]]
        
        return total
    
    def get_total_time(self, path: List[int], include_service: bool = True) -> float:
        """
        Calculate total time for a path including travel and service.
        
        Args:
            path: List of location indices
            include_service: Whether to include service times
            
        Returns:
            Total time in minutes
        """
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            total += self.time_matrix[path[i], path[i + 1]]
            if include_service and i > 0:  # Skip depot service time
                total += self.locations[path[i]].service_time
        
        return total
    
    @classmethod
    def from_coordinates(
        cls,
        coordinates: List[Tuple[float, float]],
        method: str = "euclidean",
    ) -> 'DistanceMatrix':
        """
        Create distance matrix from a list of coordinates.
        
        Args:
            coordinates: List of (x, y) or (lat, lon) tuples
            method: Distance calculation method
            
        Returns:
            DistanceMatrix instance
        """
        locations = [
            Location(id=str(i), x=coord[0], y=coord[1])
            for i, coord in enumerate(coordinates)
        ]
        return cls(locations=locations, distance_method=method)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "locations": [
                {
                    "id": loc.id,
                    "name": loc.name,
                    "coordinates": loc.coordinates,
                    "service_time": loc.service_time,
                    "demand": loc.demand,
                }
                for loc in self.locations
            ],
            "distance_method": self.distance_method,
            "matrix": self.matrix.tolist(),
        }
