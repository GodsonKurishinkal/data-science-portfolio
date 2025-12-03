"""Box/Item representation for 3D bin packing."""
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import uuid


class BoxType(Enum):
    """Types of boxes with different handling requirements."""
    STANDARD = "standard"
    FRAGILE = "fragile"
    HAZARDOUS = "hazardous"
    PERISHABLE = "perishable"
    HEAVY = "heavy"


@dataclass
class Dimensions:
    """3D dimensions of a box or container."""
    length: float  # x-axis (depth into truck)
    width: float   # y-axis (side to side)
    height: float  # z-axis (floor to ceiling)
    
    @property
    def volume(self) -> float:
        """Calculate volume."""
        return self.length * self.width * self.height
    
    def fits_in(self, other: 'Dimensions') -> bool:
        """Check if this dimension fits inside another."""
        return (self.length <= other.length and 
                self.width <= other.width and 
                self.height <= other.height)
    
    def __iter__(self):
        """Allow unpacking."""
        return iter([self.length, self.width, self.height])


@dataclass
class Position:
    """3D position in the container."""
    x: float = 0.0  # depth position
    y: float = 0.0  # width position
    z: float = 0.0  # height position
    
    def __iter__(self):
        """Allow unpacking."""
        return iter([self.x, self.y, self.z])
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Return as tuple."""
        return (self.x, self.y, self.z)


@dataclass
class Box:
    """
    Represents a package/item to be packed in a delivery vehicle.
    
    Attributes:
        id: Unique identifier for the box
        length: Length in cm (x-axis, depth)
        width: Width in cm (y-axis, side to side)
        height: Height in cm (z-axis, floor to ceiling)
        weight: Weight in kg
        sequence: Delivery sequence (1 = first delivery, higher = later)
        box_type: Type of box (affects handling)
        can_rotate: Whether the box can be rotated
        max_stack_weight: Maximum weight that can be placed on top
        destination: Delivery destination identifier
        priority: Priority level (higher = more important)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    weight: float = 0.0
    sequence: int = 0
    box_type: BoxType = BoxType.STANDARD
    can_rotate: bool = True
    max_stack_weight: float = 100.0
    destination: str = ""
    priority: int = 1
    
    # Placement information (set during packing)
    position: Position = field(default_factory=Position)
    rotated: bool = False
    packed: bool = False
    container_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate box dimensions."""
        if self.length < 0 or self.width < 0 or self.height < 0:
            raise ValueError("Box dimensions must be non-negative")
        if self.weight < 0:
            raise ValueError("Box weight must be non-negative")
    
    @property
    def dimensions(self) -> Dimensions:
        """Get current dimensions (considering rotation)."""
        if self.rotated:
            return Dimensions(self.width, self.length, self.height)
        return Dimensions(self.length, self.width, self.height)
    
    @property
    def original_dimensions(self) -> Dimensions:
        """Get original dimensions (ignoring rotation)."""
        return Dimensions(self.length, self.width, self.height)
    
    @property
    def volume(self) -> float:
        """Calculate box volume in cubic cm."""
        return self.length * self.width * self.height
    
    @property
    def density(self) -> float:
        """Calculate density (weight/volume)."""
        if self.volume == 0:
            return 0.0
        return self.weight / self.volume
    
    @property
    def base_area(self) -> float:
        """Calculate base area (length x width)."""
        dims = self.dimensions
        return dims.length * dims.width
    
    @property
    def is_fragile(self) -> bool:
        """Check if box is fragile."""
        return self.box_type == BoxType.FRAGILE
    
    @property
    def is_heavy(self) -> bool:
        """Check if box is heavy (affects stacking)."""
        return self.box_type == BoxType.HEAVY or self.weight > 30
    
    def rotate(self) -> None:
        """Rotate box 90 degrees on the XY plane (swap length and width)."""
        if self.can_rotate:
            self.rotated = not self.rotated
    
    def get_rotations(self) -> list['Box']:
        """Get all valid rotation variants of this box."""
        rotations = [self.copy()]
        if self.can_rotate and self.length != self.width:
            rotated = self.copy()
            rotated.rotated = True
            rotations.append(rotated)
        return rotations
    
    def copy(self) -> 'Box':
        """Create a copy of this box."""
        return Box(
            id=self.id,
            length=self.length,
            width=self.width,
            height=self.height,
            weight=self.weight,
            sequence=self.sequence,
            box_type=self.box_type,
            can_rotate=self.can_rotate,
            max_stack_weight=self.max_stack_weight,
            destination=self.destination,
            priority=self.priority,
            position=Position(self.position.x, self.position.y, self.position.z),
            rotated=self.rotated,
            packed=self.packed,
            container_id=self.container_id,
        )
    
    def place_at(self, position: Position, container_id: str) -> None:
        """Place the box at a specific position in a container."""
        self.position = position
        self.container_id = container_id
        self.packed = True
    
    def get_corners(self) -> list[Tuple[float, float, float]]:
        """Get all 8 corners of the placed box."""
        dims = self.dimensions
        x, y, z = self.position.x, self.position.y, self.position.z
        return [
            (x, y, z),
            (x + dims.length, y, z),
            (x, y + dims.width, z),
            (x + dims.length, y + dims.width, z),
            (x, y, z + dims.height),
            (x + dims.length, y, z + dims.height),
            (x, y + dims.width, z + dims.height),
            (x + dims.length, y + dims.width, z + dims.height),
        ]
    
    def intersects(self, other: 'Box') -> bool:
        """Check if this box intersects with another placed box."""
        if not self.packed or not other.packed:
            return False
        
        dims1 = self.dimensions
        dims2 = other.dimensions
        
        # Check for separation on each axis
        x_sep = (self.position.x + dims1.length <= other.position.x or
                 other.position.x + dims2.length <= self.position.x)
        y_sep = (self.position.y + dims1.width <= other.position.y or
                 other.position.y + dims2.width <= self.position.y)
        z_sep = (self.position.z + dims1.height <= other.position.z or
                 other.position.z + dims2.height <= self.position.z)
        
        return not (x_sep or y_sep or z_sep)
    
    def supports(self, other: 'Box') -> float:
        """
        Calculate the support area this box provides to another box on top.
        Returns the overlapping base area.
        """
        if not self.packed or not other.packed:
            return 0.0
        
        dims1 = self.dimensions
        dims2 = other.dimensions
        
        # Check if other box is on top of this one
        if abs(self.position.z + dims1.height - other.position.z) > 0.01:
            return 0.0
        
        # Calculate overlapping area
        x_overlap = max(0, min(self.position.x + dims1.length, other.position.x + dims2.length) -
                       max(self.position.x, other.position.x))
        y_overlap = max(0, min(self.position.y + dims1.width, other.position.y + dims2.width) -
                       max(self.position.y, other.position.y))
        
        return x_overlap * y_overlap
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Box(id='{self.id}', dims={self.length}x{self.width}x{self.height}, "
                f"weight={self.weight}kg, seq={self.sequence})")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "sequence": self.sequence,
            "box_type": self.box_type.value,
            "can_rotate": self.can_rotate,
            "max_stack_weight": self.max_stack_weight,
            "destination": self.destination,
            "priority": self.priority,
            "position": self.position.as_tuple() if self.packed else None,
            "rotated": self.rotated,
            "packed": self.packed,
            "container_id": self.container_id,
            "volume": self.volume,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Box':
        """Create Box from dictionary."""
        box = cls(
            id=data.get("id", ""),
            length=data.get("length", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            weight=data.get("weight", 0),
            sequence=data.get("sequence", 0),
            box_type=BoxType(data.get("box_type", "standard")),
            can_rotate=data.get("can_rotate", True),
            max_stack_weight=data.get("max_stack_weight", 100),
            destination=data.get("destination", ""),
            priority=data.get("priority", 1),
        )
        if data.get("position"):
            box.position = Position(*data["position"])
        box.rotated = data.get("rotated", False)
        box.packed = data.get("packed", False)
        box.container_id = data.get("container_id")
        return box
