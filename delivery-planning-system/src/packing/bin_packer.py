"""
3D Bin Packing Algorithm Implementation.

This module implements the Extreme Points algorithm for 3D bin packing,
with enhancements for:
- Delivery sequence optimization (LIFO loading)
- Weight distribution balancing
- Fragile item handling
- Support surface requirements
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .box import Box, Position
from .container import Container, ExtremePoint


class PackingStrategy(Enum):
    """Packing strategies for box placement."""
    BEST_FIT = "best_fit"           # Minimize wasted space
    FIRST_FIT = "first_fit"         # Use first valid position
    SEQUENCE_AWARE = "sequence_aware"  # Optimize for delivery sequence (LIFO)
    WEIGHT_BALANCED = "weight_balanced"  # Balance weight distribution


class SortingCriterion(Enum):
    """Criteria for sorting boxes before packing."""
    VOLUME_DESC = "volume_desc"       # Largest first
    WEIGHT_DESC = "weight_desc"       # Heaviest first
    HEIGHT_DESC = "height_desc"       # Tallest first
    BASE_AREA_DESC = "base_area_desc" # Largest base first
    SEQUENCE_ASC = "sequence_asc"     # By delivery sequence
    PRIORITY_DESC = "priority_desc"   # Highest priority first


@dataclass
class PackingResult:
    """
    Result of a bin packing operation.
    
    Attributes:
        success: Whether all boxes were packed
        packed_boxes: List of successfully packed boxes
        unpacked_boxes: List of boxes that couldn't be packed
        container: The container with packed boxes
        utilization: Volume utilization percentage
        weight_utilization: Weight utilization percentage
        iterations: Number of algorithm iterations
    """
    success: bool = False
    packed_boxes: List[Box] = field(default_factory=list)
    unpacked_boxes: List[Box] = field(default_factory=list)
    container: Optional[Container] = None
    utilization: float = 0.0
    weight_utilization: float = 0.0
    iterations: int = 0
    packing_sequence: List[dict] = field(default_factory=list)
    
    @property
    def num_packed(self) -> int:
        """Number of boxes packed."""
        return len(self.packed_boxes)
    
    @property
    def num_unpacked(self) -> int:
        """Number of boxes not packed."""
        return len(self.unpacked_boxes)
    
    @property
    def total_boxes(self) -> int:
        """Total number of boxes."""
        return self.num_packed + self.num_unpacked
    
    @property
    def packing_ratio(self) -> float:
        """Ratio of boxes successfully packed."""
        if self.total_boxes == 0:
            return 0.0
        return self.num_packed / self.total_boxes
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "packed_boxes": [box.to_dict() for box in self.packed_boxes],
            "unpacked_boxes": [box.to_dict() for box in self.unpacked_boxes],
            "container": self.container.to_dict() if self.container else None,
            "utilization": self.utilization,
            "weight_utilization": self.weight_utilization,
            "iterations": self.iterations,
            "num_packed": self.num_packed,
            "num_unpacked": self.num_unpacked,
            "packing_ratio": self.packing_ratio,
            "packing_sequence": self.packing_sequence,
        }


class BinPacker:
    """
    3D Bin Packing using the Extreme Points algorithm.
    
    The algorithm works by:
    1. Starting with an empty container with initial extreme point at origin
    2. For each box (sorted by size/priority):
       a. Find all valid extreme points where the box can fit
       b. Select the best position based on the packing strategy
       c. Place the box and generate new extreme points
    3. Continue until all boxes are packed or no valid positions remain
    
    For delivery sequence optimization (LIFO), boxes are sorted so that
    items delivered first are loaded last (closest to the door).
    """
    
    def __init__(
        self,
        strategy: PackingStrategy = PackingStrategy.SEQUENCE_AWARE,
        sorting: SortingCriterion = SortingCriterion.VOLUME_DESC,
        allow_rotation: bool = True,
        min_support_ratio: float = 0.7,
        max_iterations: int = 10000,
    ):
        """
        Initialize the bin packer.
        
        Args:
            strategy: Packing strategy to use
            sorting: Criterion for sorting boxes before packing
            allow_rotation: Whether to allow box rotation
            min_support_ratio: Minimum required support for stacked boxes
            max_iterations: Maximum algorithm iterations
        """
        self.strategy = strategy
        self.sorting = sorting
        self.allow_rotation = allow_rotation
        self.min_support_ratio = min_support_ratio
        self.max_iterations = max_iterations
    
    def pack(
        self,
        boxes: List[Box],
        container: Container,
        optimize_sequence: bool = True,
    ) -> PackingResult:
        """
        Pack boxes into a container.
        
        Args:
            boxes: List of boxes to pack
            container: The container to pack into
            optimize_sequence: Whether to optimize for delivery sequence
            
        Returns:
            PackingResult with packing details
        """
        # Reset container
        container.clear()
        
        # Create copies of boxes to avoid modifying originals
        boxes_to_pack = [box.copy() for box in boxes]
        
        # Sort boxes based on strategy
        if optimize_sequence and self.strategy == PackingStrategy.SEQUENCE_AWARE:
            # For LIFO: pack in reverse delivery sequence (last delivery first)
            boxes_to_pack = self._sort_for_lifo(boxes_to_pack)
        else:
            boxes_to_pack = self._sort_boxes(boxes_to_pack)
        
        # Packing loop
        packed_boxes = []
        unpacked_boxes = []
        packing_sequence = []
        iterations = 0
        
        for box in boxes_to_pack:
            if iterations >= self.max_iterations:
                unpacked_boxes.append(box)
                continue
            
            # Find best position for this box
            best_position = self._find_best_position(box, container)
            
            if best_position is not None:
                # Place the box
                if container.place_box(box, best_position):
                    packed_boxes.append(box)
                    packing_sequence.append({
                        "box_id": box.id,
                        "position": best_position.as_tuple(),
                        "rotated": box.rotated,
                        "dimensions": [box.dimensions.length, 
                                       box.dimensions.width, 
                                       box.dimensions.height],
                        "sequence": box.sequence,
                        "step": len(packed_boxes),
                    })
                else:
                    unpacked_boxes.append(box)
            else:
                unpacked_boxes.append(box)
            
            iterations += 1
        
        # Build result
        result = PackingResult(
            success=len(unpacked_boxes) == 0,
            packed_boxes=packed_boxes,
            unpacked_boxes=unpacked_boxes,
            container=container,
            utilization=container.volume_utilization,
            weight_utilization=container.weight_utilization,
            iterations=iterations,
            packing_sequence=packing_sequence,
        )
        
        return result
    
    def pack_multiple_containers(
        self,
        boxes: List[Box],
        containers: List[Container],
    ) -> List[PackingResult]:
        """
        Pack boxes into multiple containers.
        
        Args:
            boxes: List of boxes to pack
            containers: List of available containers
            
        Returns:
            List of PackingResult for each container used
        """
        results = []
        remaining_boxes = [box.copy() for box in boxes]
        
        for container in containers:
            if not remaining_boxes:
                break
            
            result = self.pack(remaining_boxes, container)
            results.append(result)
            
            # Update remaining boxes
            remaining_boxes = [box.copy() for box in result.unpacked_boxes]
        
        return results
    
    def _sort_boxes(self, boxes: List[Box]) -> List[Box]:
        """Sort boxes based on the sorting criterion."""
        key_funcs = {
            SortingCriterion.VOLUME_DESC: lambda b: -b.volume,
            SortingCriterion.WEIGHT_DESC: lambda b: -b.weight,
            SortingCriterion.HEIGHT_DESC: lambda b: -b.height,
            SortingCriterion.BASE_AREA_DESC: lambda b: -b.base_area,
            SortingCriterion.SEQUENCE_ASC: lambda b: b.sequence,
            SortingCriterion.PRIORITY_DESC: lambda b: -b.priority,
        }
        
        key_func = key_funcs.get(self.sorting, lambda b: -b.volume)
        return sorted(boxes, key=key_func)
    
    def _sort_for_lifo(self, boxes: List[Box]) -> List[Box]:
        """
        Sort boxes for LIFO (Last In, First Out) loading.
        
        Items delivered first should be loaded last (near the door).
        Within same sequence, larger/heavier items go first.
        """
        def lifo_key(box: Box):
            # Reverse sequence (higher sequence = earlier in packing)
            # Then by volume descending, then weight descending
            return (-box.sequence, -box.volume, -box.weight)
        
        return sorted(boxes, key=lifo_key)
    
    def _find_best_position(
        self,
        box: Box,
        container: Container,
    ) -> Optional[Position]:
        """
        Find the best position for a box in the container.
        
        Args:
            box: The box to place
            container: The container to place in
            
        Returns:
            Best position or None if no valid position found
        """
        best_position = None
        best_score = float('inf')
        best_rotation = False
        
        # Try each extreme point
        for ep in container.extreme_points:
            # Try both orientations if rotation allowed
            rotations = box.get_rotations() if self.allow_rotation else [box]
            
            for rotated_box in rotations:
                position = ep.as_position()
                
                # Check if box fits at this position
                if not container.can_fit_box(rotated_box, position):
                    continue
                
                # Check support requirements
                if position.z > 0:
                    support = container.get_support_at_position(rotated_box, position)
                    if support < self.min_support_ratio:
                        continue
                
                # Check fragile/stacking constraints
                if not self._check_stacking_constraints(rotated_box, position, container):
                    continue
                
                # Calculate score for this position
                score = self._calculate_position_score(
                    rotated_box, position, container, ep
                )
                
                if score < best_score:
                    best_score = score
                    best_position = position
                    best_rotation = rotated_box.rotated
        
        # Apply rotation to original box if needed
        if best_position is not None and best_rotation != box.rotated:
            box.rotate()
        
        return best_position
    
    def _calculate_position_score(
        self,
        box: Box,
        position: Position,
        container: Container,
        extreme_point: ExtremePoint,
    ) -> float:
        """
        Calculate a score for placing a box at a position.
        Lower scores are better.
        """
        score = 0.0
        
        if self.strategy == PackingStrategy.BEST_FIT:
            # Minimize wasted space at this position
            dims = box.dimensions
            wasted_x = extreme_point.max_length - dims.length
            wasted_y = extreme_point.max_width - dims.width
            wasted_z = extreme_point.max_height - dims.height
            score = wasted_x * wasted_y * wasted_z
        
        elif self.strategy == PackingStrategy.FIRST_FIT:
            # Just use position order (extreme points are sorted)
            score = position.x + position.y + position.z
        
        elif self.strategy == PackingStrategy.SEQUENCE_AWARE:
            # For LIFO: prefer positions further from door (back of truck)
            # Door is at x = container.length (front)
            # Higher sequence items should be at lower x values
            door_distance = container.length - (position.x + box.dimensions.length)
            
            # Items delivered last (high sequence) should be far from door
            # Items delivered first (low sequence) should be near door
            sequence_factor = box.sequence / max(1, max(b.sequence for b in container.packed_boxes + [box]))
            
            # Prefer back positions for late delivery items
            score = -door_distance * (1 - sequence_factor) + door_distance * sequence_factor
            
            # Also prefer lower positions (bottom first)
            score += position.z * 0.01
        
        elif self.strategy == PackingStrategy.WEIGHT_BALANCED:
            # Consider weight distribution
            dist = container.get_weight_distribution()
            center_x = container.length / 2
            center_y = container.width / 2
            box_center_x = position.x + box.dimensions.length / 2
            box_center_y = position.y + box.dimensions.width / 2
            
            # Prefer positions that balance weight
            if dist["front"] > dist["back"]:
                score += (box_center_x - center_x) * box.weight
            else:
                score += (center_x - box_center_x) * box.weight
            
            if dist["left"] > dist["right"]:
                score += (box_center_y - center_y) * box.weight
            else:
                score += (center_y - box_center_y) * box.weight
        
        return score
    
    def _check_stacking_constraints(
        self,
        box: Box,
        position: Position,
        container: Container,
    ) -> bool:
        """
        Check stacking constraints for a box placement.
        
        - Fragile items shouldn't have heavy items on top
        - Weight limits for stacking
        - Heavy items should go on bottom
        """
        # Check if placing on top of other boxes
        if position.z > 0:
            # Find supporting boxes
            for packed_box in container.packed_boxes:
                overlap = self._calculate_overlap(box, position, packed_box)
                if overlap > 0:
                    # Check if this box is too heavy for the supporting box
                    if box.weight > packed_box.max_stack_weight:
                        return False
                    # Don't stack on fragile items
                    if packed_box.is_fragile:
                        return False
        
        # Fragile items should ideally be on top
        # (This is a soft constraint - we allow it but prefer top positions)
        
        return True
    
    def _calculate_overlap(
        self,
        box: Box,
        position: Position,
        other_box: Box,
    ) -> float:
        """Calculate the overlapping base area between two boxes."""
        dims1 = box.dimensions
        dims2 = other_box.dimensions
        
        x_overlap = max(0, min(position.x + dims1.length, 
                               other_box.position.x + dims2.length) -
                       max(position.x, other_box.position.x))
        y_overlap = max(0, min(position.y + dims1.width,
                               other_box.position.y + dims2.width) -
                       max(position.y, other_box.position.y))
        
        return x_overlap * y_overlap


def create_packer(config: dict = None) -> BinPacker:
    """
    Factory function to create a BinPacker from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BinPacker instance
    """
    if config is None:
        config = {}
    
    strategy = PackingStrategy(config.get("strategy", "sequence_aware"))
    sorting = SortingCriterion(config.get("sorting", "volume_desc"))
    
    return BinPacker(
        strategy=strategy,
        sorting=sorting,
        allow_rotation=config.get("allow_rotation", True),
        min_support_ratio=config.get("min_support_ratio", 0.7),
        max_iterations=config.get("max_iterations", 10000),
    )
