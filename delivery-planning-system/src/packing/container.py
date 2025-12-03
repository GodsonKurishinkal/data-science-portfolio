"""Container/Vehicle representation for 3D bin packing."""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import uuid

from .box import Box, Dimensions, Position


class ContainerType(Enum):
    """Types of delivery vehicles/containers."""
    SMALL_VAN = "small_van"
    MEDIUM_TRUCK = "medium_truck"
    LARGE_TRUCK = "large_truck"
    SEMI_TRAILER = "semi_trailer"
    CUSTOM = "custom"


# Predefined container dimensions (cm) and weight limits (kg)
CONTAINER_SPECS = {
    ContainerType.SMALL_VAN: {
        "length": 300, "width": 170, "height": 180, "max_weight": 1000
    },
    ContainerType.MEDIUM_TRUCK: {
        "length": 450, "width": 220, "height": 230, "max_weight": 5000
    },
    ContainerType.LARGE_TRUCK: {
        "length": 600, "width": 250, "height": 270, "max_weight": 10000
    },
    ContainerType.SEMI_TRAILER: {
        "length": 1360, "width": 250, "height": 270, "max_weight": 25000
    },
}


@dataclass
class ExtremePoint:
    """
    Represents an extreme point (potential placement position) in the container.
    
    Extreme points are the corners of placed boxes where new boxes can potentially
    be placed. This is the core of the Extreme Points algorithm.
    """
    x: float
    y: float
    z: float
    
    # Maximum dimensions available at this point
    max_length: float = float('inf')
    max_width: float = float('inf')
    max_height: float = float('inf')
    
    # Supporting surface information
    support_area: float = 0.0
    support_ratio: float = 0.0  # Percentage of box base that would be supported
    
    def as_position(self) -> Position:
        """Convert to Position."""
        return Position(self.x, self.y, self.z)
    
    def can_fit(self, dims: Dimensions) -> bool:
        """Check if a box with given dimensions can fit at this point."""
        return (dims.length <= self.max_length and
                dims.width <= self.max_width and
                dims.height <= self.max_height)
    
    def __lt__(self, other: 'ExtremePoint') -> bool:
        """Comparison for sorting (prefer lower, then left-back)."""
        if self.z != other.z:
            return self.z < other.z
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y
    
    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if not isinstance(other, ExtremePoint):
            return False
        return (abs(self.x - other.x) < 0.01 and
                abs(self.y - other.y) < 0.01 and
                abs(self.z - other.z) < 0.01)
    
    def __hash__(self) -> int:
        """Hash for set operations."""
        return hash((round(self.x, 2), round(self.y, 2), round(self.z, 2)))


@dataclass
class Container:
    """
    Represents a delivery vehicle or container for packing boxes.
    
    Attributes:
        id: Unique identifier
        length: Internal length in cm (x-axis, depth)
        width: Internal width in cm (y-axis, side to side)
        height: Internal height in cm (z-axis, floor to ceiling)
        max_weight: Maximum load capacity in kg
        container_type: Type of container
        vehicle_id: Associated vehicle ID (if any)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    length: float = 600.0
    width: float = 250.0
    height: float = 270.0
    max_weight: float = 10000.0
    container_type: ContainerType = ContainerType.LARGE_TRUCK
    vehicle_id: Optional[str] = None
    
    # Packing state
    packed_boxes: List[Box] = field(default_factory=list)
    extreme_points: List[ExtremePoint] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize extreme points with origin."""
        if not self.extreme_points:
            self.extreme_points = [
                ExtremePoint(0, 0, 0, self.length, self.width, self.height)
            ]
    
    @classmethod
    def from_type(cls, container_type: ContainerType, 
                  container_id: Optional[str] = None) -> 'Container':
        """Create container from predefined type."""
        specs = CONTAINER_SPECS.get(container_type, CONTAINER_SPECS[ContainerType.LARGE_TRUCK])
        return cls(
            id=container_id or str(uuid.uuid4())[:8],
            length=specs["length"],
            width=specs["width"],
            height=specs["height"],
            max_weight=specs["max_weight"],
            container_type=container_type,
        )
    
    @property
    def dimensions(self) -> Dimensions:
        """Get container dimensions."""
        return Dimensions(self.length, self.width, self.height)
    
    @property
    def volume(self) -> float:
        """Get total container volume."""
        return self.length * self.width * self.height
    
    @property
    def used_volume(self) -> float:
        """Calculate total volume used by packed boxes."""
        return sum(box.volume for box in self.packed_boxes)
    
    @property
    def remaining_volume(self) -> float:
        """Calculate remaining volume."""
        return self.volume - self.used_volume
    
    @property
    def volume_utilization(self) -> float:
        """Calculate volume utilization percentage."""
        if self.volume == 0:
            return 0.0
        return self.used_volume / self.volume
    
    @property
    def current_weight(self) -> float:
        """Calculate current total weight of packed boxes."""
        return sum(box.weight for box in self.packed_boxes)
    
    @property
    def remaining_weight_capacity(self) -> float:
        """Calculate remaining weight capacity."""
        return self.max_weight - self.current_weight
    
    @property
    def weight_utilization(self) -> float:
        """Calculate weight utilization percentage."""
        if self.max_weight == 0:
            return 0.0
        return self.current_weight / self.max_weight
    
    @property
    def num_boxes(self) -> int:
        """Get number of packed boxes."""
        return len(self.packed_boxes)
    
    def can_fit_weight(self, weight: float) -> bool:
        """Check if additional weight can be added."""
        return self.current_weight + weight <= self.max_weight
    
    def can_fit_box(self, box: Box, position: Position) -> bool:
        """
        Check if a box can be placed at the given position.
        
        Args:
            box: The box to place
            position: The target position
            
        Returns:
            True if the box fits without overlapping and within bounds
        """
        dims = box.dimensions
        
        # Check container bounds
        if (position.x + dims.length > self.length or
            position.y + dims.width > self.width or
            position.z + dims.height > self.height):
            return False
        
        # Check weight capacity
        if not self.can_fit_weight(box.weight):
            return False
        
        # Create temporary box for intersection check
        temp_box = box.copy()
        temp_box.position = position
        temp_box.packed = True
        
        # Check for intersections with existing boxes
        for packed_box in self.packed_boxes:
            if temp_box.intersects(packed_box):
                return False
        
        return True
    
    def place_box(self, box: Box, position: Position) -> bool:
        """
        Place a box at the specified position.
        
        Args:
            box: The box to place
            position: The target position
            
        Returns:
            True if successfully placed, False otherwise
        """
        if not self.can_fit_box(box, position):
            return False
        
        # Update box placement
        box.place_at(position, self.id)
        self.packed_boxes.append(box)
        
        # Update extreme points
        self._update_extreme_points(box)
        
        return True
    
    def _update_extreme_points(self, placed_box: Box) -> None:
        """
        Update extreme points after placing a box.
        
        The Extreme Points algorithm generates new potential placement points
        at the corners of the newly placed box.
        """
        dims = placed_box.dimensions
        pos = placed_box.position
        
        # Remove extreme points that are now inside the placed box
        valid_points = []
        for ep in self.extreme_points:
            if not self._point_inside_box(ep, placed_box):
                valid_points.append(ep)
        
        # Generate new extreme points from the placed box
        new_points = [
            # Right face (x + length)
            ExtremePoint(pos.x + dims.length, pos.y, pos.z),
            # Front face (y + width)
            ExtremePoint(pos.x, pos.y + dims.width, pos.z),
            # Top face (z + height)
            ExtremePoint(pos.x, pos.y, pos.z + dims.height),
        ]
        
        # Add valid new points
        for np in new_points:
            if self._is_valid_extreme_point(np):
                # Calculate available space at this point
                np.max_length = self.length - np.x
                np.max_width = self.width - np.y
                np.max_height = self.height - np.z
                
                # Update based on existing boxes
                for box in self.packed_boxes:
                    self._update_point_limits(np, box)
                
                # Add if not duplicate
                if np not in valid_points:
                    valid_points.append(np)
        
        # Sort extreme points (prefer lower, then back-left)
        self.extreme_points = sorted(valid_points)
    
    def _point_inside_box(self, point: ExtremePoint, box: Box) -> bool:
        """Check if a point is inside a box."""
        dims = box.dimensions
        return (box.position.x <= point.x < box.position.x + dims.length and
                box.position.y <= point.y < box.position.y + dims.width and
                box.position.z <= point.z < box.position.z + dims.height)
    
    def _is_valid_extreme_point(self, point: ExtremePoint) -> bool:
        """Check if an extreme point is valid (within container bounds)."""
        return (0 <= point.x < self.length and
                0 <= point.y < self.width and
                0 <= point.z < self.height)
    
    def _update_point_limits(self, point: ExtremePoint, box: Box) -> None:
        """Update available space limits at an extreme point based on a box."""
        dims = box.dimensions
        
        # Check if box blocks space in each direction
        if (box.position.y <= point.y < box.position.y + dims.width and
            box.position.z <= point.z < box.position.z + dims.height and
            box.position.x > point.x):
            point.max_length = min(point.max_length, box.position.x - point.x)
        
        if (box.position.x <= point.x < box.position.x + dims.length and
            box.position.z <= point.z < box.position.z + dims.height and
            box.position.y > point.y):
            point.max_width = min(point.max_width, box.position.y - point.y)
        
        if (box.position.x <= point.x < box.position.x + dims.length and
            box.position.y <= point.y < box.position.y + dims.width and
            box.position.z > point.z):
            point.max_height = min(point.max_height, box.position.z - point.z)
    
    def get_support_at_position(self, box: Box, position: Position) -> float:
        """
        Calculate the support ratio for a box at a given position.
        
        Returns the percentage of the box's base area that is supported
        by either the container floor or other boxes.
        """
        dims = box.dimensions
        base_area = dims.length * dims.width
        
        if base_area == 0:
            return 0.0
        
        # If on the floor, fully supported
        if position.z == 0:
            return 1.0
        
        # Calculate support from boxes below
        total_support = 0.0
        for packed_box in self.packed_boxes:
            packed_dims = packed_box.dimensions
            
            # Check if this box could support the new box
            if abs(packed_box.position.z + packed_dims.height - position.z) > 0.01:
                continue
            
            # Calculate overlapping area
            x_overlap = max(0, min(packed_box.position.x + packed_dims.length, 
                                   position.x + dims.length) -
                           max(packed_box.position.x, position.x))
            y_overlap = max(0, min(packed_box.position.y + packed_dims.width,
                                   position.y + dims.width) -
                           max(packed_box.position.y, position.y))
            
            total_support += x_overlap * y_overlap
        
        return min(1.0, total_support / base_area)
    
    def get_weight_distribution(self) -> dict:
        """
        Calculate weight distribution in the container.
        
        Returns distribution across front/back and left/right halves.
        """
        front_weight = 0.0
        back_weight = 0.0
        left_weight = 0.0
        right_weight = 0.0
        
        mid_x = self.length / 2
        mid_y = self.width / 2
        
        for box in self.packed_boxes:
            dims = box.dimensions
            center_x = box.position.x + dims.length / 2
            center_y = box.position.y + dims.width / 2
            
            if center_x < mid_x:
                back_weight += box.weight
            else:
                front_weight += box.weight
            
            if center_y < mid_y:
                left_weight += box.weight
            else:
                right_weight += box.weight
        
        total = self.current_weight or 1.0  # Avoid division by zero
        
        return {
            "front": front_weight,
            "back": back_weight,
            "left": left_weight,
            "right": right_weight,
            "front_ratio": front_weight / total,
            "back_ratio": back_weight / total,
            "left_ratio": left_weight / total,
            "right_ratio": right_weight / total,
            "front_back_balance": abs(front_weight - back_weight) / total,
            "left_right_balance": abs(left_weight - right_weight) / total,
        }
    
    def get_delivery_sequence(self) -> List[Box]:
        """
        Get boxes in delivery sequence order.
        
        For LIFO (Last In, First Out) unloading, boxes delivered first
        should be loaded last (closest to door).
        """
        return sorted(self.packed_boxes, key=lambda b: b.sequence)
    
    def clear(self) -> None:
        """Clear all packed boxes and reset extreme points."""
        for box in self.packed_boxes:
            box.packed = False
            box.container_id = None
            box.position = Position()
        
        self.packed_boxes = []
        self.extreme_points = [
            ExtremePoint(0, 0, 0, self.length, self.width, self.height)
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_weight": self.max_weight,
            "container_type": self.container_type.value,
            "vehicle_id": self.vehicle_id,
            "packed_boxes": [box.to_dict() for box in self.packed_boxes],
            "volume": self.volume,
            "used_volume": self.used_volume,
            "volume_utilization": self.volume_utilization,
            "current_weight": self.current_weight,
            "weight_utilization": self.weight_utilization,
            "num_boxes": self.num_boxes,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Container(id='{self.id}', dims={self.length}x{self.width}x{self.height}, "
                f"boxes={self.num_boxes}, util={self.volume_utilization:.1%})")
