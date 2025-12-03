"""
Tests for the 3D Bin Packing module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from packing.box import Box, BoxType, Dimensions, Position
from packing.container import Container, ContainerType, ExtremePoint
from packing.bin_packer import BinPacker, PackingStrategy, SortingCriterion


class TestBox:
    """Tests for Box class."""
    
    def test_box_creation(self):
        """Test basic box creation."""
        box = Box(
            id="TEST-001",
            length=50,
            width=40,
            height=30,
            weight=10,
            box_type=BoxType.STANDARD
        )
        assert box.id == "TEST-001"
        assert box.length == 50
        assert box.width == 40
        assert box.height == 30
        assert box.weight == 10
    
    def test_box_volume(self):
        """Test volume calculation."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5)
        assert box.volume == 10 * 20 * 30
    
    def test_box_dimensions(self):
        """Test dimensions property."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5)
        dims = box.dimensions
        assert dims.length == 10
        assert dims.width == 20
        assert dims.height == 30
    
    def test_box_rotation(self):
        """Test box rotation."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5)
        box.rotate()
        # Rotation swaps rotated flag, dimensions property reflects this
        dims = box.dimensions
        assert dims.length == 20  # Swapped
        assert dims.width == 10   # Swapped
        assert dims.height == 30  # Unchanged
    
    def test_box_position(self):
        """Test setting box position."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5)
        box.place_at(Position(100, 50, 0), "CONTAINER-1")
        assert box.position is not None
        assert box.position.x == 100
        assert box.position.y == 50
        assert box.position.z == 0
        assert box.packed is True
        assert box.container_id == "CONTAINER-1"
    
    def test_fragile_box(self):
        """Test fragile box property."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5, box_type=BoxType.FRAGILE)
        assert box.is_fragile is True
    
    def test_delivery_sequence(self):
        """Test delivery sequence."""
        box = Box(id="B1", length=10, width=20, height=30, weight=5, sequence=3)
        assert box.sequence == 3


class TestDimensions:
    """Tests for Dimensions class."""
    
    def test_dimensions_creation(self):
        """Test dimensions creation."""
        dims = Dimensions(100, 50, 30)
        assert dims.length == 100
        assert dims.width == 50
        assert dims.height == 30
    
    def test_dimensions_volume(self):
        """Test volume calculation."""
        dims = Dimensions(10, 20, 30)
        assert dims.volume == 6000


class TestPosition:
    """Tests for Position class."""
    
    def test_position_creation(self):
        """Test position creation."""
        pos = Position(10, 20, 30)
        assert pos.x == 10
        assert pos.y == 20
        assert pos.z == 30


class TestContainer:
    """Tests for Container class."""
    
    def test_container_creation(self):
        """Test container creation."""
        container = Container(
            id="TRUCK-001",
            container_type=ContainerType.LARGE_TRUCK,
            length=600,
            width=250,
            height=270,
            max_weight=5000
        )
        assert container.id == "TRUCK-001"
        assert container.length == 600
    
    def test_container_volume(self):
        """Test container volume."""
        container = Container(id="C1", container_type=ContainerType.SMALL_VAN, 
                             length=300, width=150, height=180, max_weight=1000)
        assert container.volume == 300 * 150 * 180
    
    def test_initial_extreme_point(self):
        """Test initial extreme point at origin."""
        container = Container(id="C1", container_type=ContainerType.SMALL_VAN, 
                             length=300, width=150, height=180, max_weight=1000)
        assert len(container.extreme_points) == 1
        ep = container.extreme_points[0]
        assert ep.x == 0 and ep.y == 0 and ep.z == 0
    
    def test_box_placement(self):
        """Test placing a box in container."""
        container = Container(id="C1", container_type=ContainerType.SMALL_VAN, 
                             length=300, width=150, height=180, max_weight=1000)
        box = Box(id="B1", length=50, width=40, height=30, weight=10)
        
        result = container.place_box(box, Position(0, 0, 0))
        assert result is True
        assert len(container.packed_boxes) == 1
    
    def test_box_fits_check(self):
        """Test if box fits at position."""
        container = Container(id="C1", container_type=ContainerType.SMALL_VAN, 
                             length=300, width=150, height=180, max_weight=1000)
        box = Box(id="B1", length=50, width=40, height=30, weight=10)
        
        # Should fit at origin
        assert container.can_fit_box(box, Position(0, 0, 0)) is True
        
        # Should not fit outside bounds
        assert container.can_fit_box(box, Position(280, 0, 0)) is False
    
    def test_utilization_calculations(self):
        """Test volume and weight utilization."""
        container = Container(id="C1", container_type=ContainerType.CUSTOM, 
                             length=100, width=100, height=100, max_weight=1000)
        box = Box(id="B1", length=50, width=50, height=50, weight=100)
        
        container.place_box(box, Position(0, 0, 0))
        
        # 50*50*50 / 100*100*100 = 0.125 = 12.5%
        assert container.volume_utilization == pytest.approx(0.125, 0.01)
        # 100 / 1000 = 0.10 = 10%
        assert container.weight_utilization == pytest.approx(0.10, 0.01)


class TestExtremePoint:
    """Tests for ExtremePoint class."""
    
    def test_extreme_point_creation(self):
        """Test extreme point creation."""
        ep = ExtremePoint(10, 20, 30)
        assert ep.x == 10
        assert ep.y == 20
        assert ep.z == 30
    
    def test_extreme_point_comparison(self):
        """Test extreme point comparison (lower z preferred)."""
        ep1 = ExtremePoint(0, 0, 0)
        ep2 = ExtremePoint(0, 0, 10)
        assert ep1 < ep2  # Lower z is preferred


class TestBinPacker:
    """Tests for BinPacker class."""
    
    def test_packer_creation(self):
        """Test packer creation with strategy."""
        packer = BinPacker(
            strategy=PackingStrategy.BEST_FIT,
            sorting=SortingCriterion.VOLUME_DESC
        )
        assert packer.strategy == PackingStrategy.BEST_FIT
    
    def test_pack_single_box(self):
        """Test packing a single box."""
        packer = BinPacker()
        container = Container(id="C1", container_type=ContainerType.SMALL_VAN, 
                             length=300, width=150, height=180, max_weight=1000)
        boxes = [Box(id="B1", length=50, width=40, height=30, weight=10)]
        
        result = packer.pack(boxes, container)
        
        assert result.num_packed == 1
        assert result.num_unpacked == 0
    
    def test_pack_multiple_boxes(self):
        """Test packing multiple boxes."""
        packer = BinPacker()
        container = Container(id="C1", container_type=ContainerType.LARGE_TRUCK, 
                             length=600, width=250, height=270, max_weight=5000)
        boxes = [
            Box(id="B1", length=50, width=40, height=30, weight=10),
            Box(id="B2", length=45, width=35, height=25, weight=8),
            Box(id="B3", length=60, width=50, height=40, weight=15),
        ]
        
        result = packer.pack(boxes, container)
        
        assert result.num_packed == 3
        assert result.num_unpacked == 0
    
    def test_pack_oversized_box(self):
        """Test that oversized box is not packed."""
        packer = BinPacker()
        container = Container(id="C1", container_type=ContainerType.CUSTOM, 
                             length=100, width=100, height=100, max_weight=1000)
        boxes = [Box(id="B1", length=200, width=200, height=200, weight=50)]  # Too big
        
        result = packer.pack(boxes, container)
        
        assert result.num_packed == 0
        assert result.num_unpacked == 1
    
    def test_pack_overweight(self):
        """Test that overweight situation is handled."""
        packer = BinPacker()
        container = Container(id="C1", container_type=ContainerType.CUSTOM, 
                             length=100, width=100, height=100, max_weight=10)  # Low weight limit
        boxes = [Box(id="B1", length=30, width=30, height=30, weight=50)]  # Heavy
        
        result = packer.pack(boxes, container)
        
        assert result.num_unpacked == 1
    
    def test_sequence_aware_packing(self):
        """Test LIFO packing for delivery sequence."""
        packer = BinPacker(
            strategy=PackingStrategy.SEQUENCE_AWARE,
            sorting=SortingCriterion.SEQUENCE_ASC
        )
        container = Container(id="C1", container_type=ContainerType.LARGE_TRUCK, 
                             length=600, width=250, height=270, max_weight=5000)
        
        # Create boxes with different sequences
        boxes = [
            Box(id="B1", length=50, width=40, height=30, weight=10, sequence=1),  # First delivery
            Box(id="B2", length=45, width=35, height=25, weight=8, sequence=2),
            Box(id="B3", length=40, width=35, height=30, weight=9, sequence=3),  # Last delivery
        ]
        
        result = packer.pack(boxes, container)
        
        # All should be packed
        assert result.num_packed == 3
    
    def test_volume_utilization_calculation(self):
        """Test volume utilization in packing result."""
        packer = BinPacker()
        container = Container(id="C1", container_type=ContainerType.CUSTOM, 
                             length=100, width=100, height=100, max_weight=1000)
        boxes = [Box(id="B1", length=50, width=50, height=50, weight=10)]
        
        result = packer.pack(boxes, container)
        
        # 50*50*50 / 100*100*100 * 100 = 12.5%
        assert result.utilization == pytest.approx(12.5, 0.5)


class TestPackingStrategies:
    """Tests for different packing strategies."""
    
    def test_best_fit_strategy(self):
        """Test best fit packing strategy."""
        packer = BinPacker(strategy=PackingStrategy.BEST_FIT)
        container = Container(id="C1", container_type=ContainerType.LARGE_TRUCK, 
                             length=600, width=250, height=270, max_weight=5000)
        boxes = [
            Box(id=f"B{i}", length=50, width=40, height=30, weight=10)
            for i in range(5)
        ]
        
        result = packer.pack(boxes, container)
        assert result.num_packed == 5
    
    def test_weight_balanced_strategy(self):
        """Test weight balanced packing strategy."""
        packer = BinPacker(strategy=PackingStrategy.WEIGHT_BALANCED)
        container = Container(id="C1", container_type=ContainerType.LARGE_TRUCK, 
                             length=600, width=250, height=270, max_weight=5000)
        boxes = [
            Box(id="B1", length=80, width=60, height=50, weight=100),  # Heavy
            Box(id="B2", length=80, width=60, height=50, weight=100),  # Heavy
            Box(id="B3", length=40, width=30, height=20, weight=5),    # Light
            Box(id="B4", length=40, width=30, height=20, weight=5),    # Light
        ]
        
        result = packer.pack(boxes, container)
        # Should pack all with weight consideration
        assert result.num_packed == 4


class TestSortingCriteria:
    """Tests for sorting criteria."""
    
    def test_volume_desc_sorting(self):
        """Test sorting by volume descending."""
        packer = BinPacker(sorting=SortingCriterion.VOLUME_DESC)
        container = Container(id="C1", container_type=ContainerType.LARGE_TRUCK, 
                             length=600, width=250, height=270, max_weight=5000)
        
        boxes = [
            Box(id="SMALL", length=20, width=20, height=20, weight=5),
            Box(id="LARGE", length=100, width=100, height=100, weight=50),
            Box(id="MEDIUM", length=50, width=50, height=50, weight=20),
        ]
        
        result = packer.pack(boxes, container)
        
        # Largest should be packed first and at the back
        assert result.num_packed == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
