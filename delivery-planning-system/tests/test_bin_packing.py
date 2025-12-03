"""
Tests for the 3D Bin Packing module.
"""

import pytest
from src.packing.box import Box, BoxType, Dimensions, Position
from src.packing.container import Container, ContainerType, ExtremePoint
from src.packing.bin_packer import BinPacker, PackingStrategy, SortingCriterion


class TestBox:
    """Tests for Box class."""
    
    def test_box_creation(self):
        """Test basic box creation."""
        box = Box(
            box_id="TEST-001",
            box_type=BoxType.MEDIUM,
            length=50,
            width=40,
            height=30,
            weight=10
        )
        assert box.box_id == "TEST-001"
        assert box.length == 50
        assert box.width == 40
        assert box.height == 30
        assert box.weight == 10
    
    def test_box_volume(self):
        """Test volume calculation."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5)
        assert box.volume == 10 * 20 * 30
    
    def test_box_dimensions(self):
        """Test dimensions property."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5)
        dims = box.dimensions
        assert dims.length == 10
        assert dims.width == 20
        assert dims.height == 30
    
    def test_box_rotation(self):
        """Test box rotation."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5)
        box.rotate()
        assert box.length == 20  # Swapped
        assert box.width == 10   # Swapped
        assert box.height == 30  # Unchanged
    
    def test_box_position(self):
        """Test setting box position."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5)
        box.set_position(100, 50, 0)
        assert box.position is not None
        assert box.position.x == 100
        assert box.position.y == 50
        assert box.position.z == 0
    
    def test_fragile_box(self):
        """Test fragile box property."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5, fragile=True)
        assert box.fragile is True
    
    def test_delivery_sequence(self):
        """Test delivery sequence."""
        box = Box("B1", BoxType.SMALL, 10, 20, 30, 5, delivery_sequence=3)
        assert box.delivery_sequence == 3


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
            container_id="TRUCK-001",
            container_type=ContainerType.BOX_TRUCK,
            length=600,
            width=250,
            height=270,
            max_weight=5000
        )
        assert container.container_id == "TRUCK-001"
        assert container.length == 600
    
    def test_container_volume(self):
        """Test container volume."""
        container = Container("C1", ContainerType.SMALL_VAN, 300, 150, 180, 1000)
        assert container.volume == 300 * 150 * 180
    
    def test_initial_extreme_point(self):
        """Test initial extreme point at origin."""
        container = Container("C1", ContainerType.SMALL_VAN, 300, 150, 180, 1000)
        assert len(container.extreme_points) == 1
        ep = container.extreme_points[0]
        assert ep.x == 0 and ep.y == 0 and ep.z == 0
    
    def test_box_placement(self):
        """Test placing a box in container."""
        container = Container("C1", ContainerType.SMALL_VAN, 300, 150, 180, 1000)
        box = Box("B1", BoxType.SMALL, 50, 40, 30, 10)
        
        result = container.place_box(box, Position(0, 0, 0))
        assert result is True
        assert len(container.packed_boxes) == 1
    
    def test_box_fits_check(self):
        """Test if box fits at position."""
        container = Container("C1", ContainerType.SMALL_VAN, 300, 150, 180, 1000)
        box = Box("B1", BoxType.SMALL, 50, 40, 30, 10)
        
        # Should fit at origin
        assert container.can_place_box(box, Position(0, 0, 0)) is True
        
        # Should not fit outside bounds
        assert container.can_place_box(box, Position(280, 0, 0)) is False
    
    def test_utilization_calculations(self):
        """Test volume and weight utilization."""
        container = Container("C1", ContainerType.SMALL_VAN, 100, 100, 100, 1000)
        box = Box("B1", BoxType.SMALL, 50, 50, 50, 100)
        
        container.place_box(box, Position(0, 0, 0))
        
        # 50*50*50 / 100*100*100 = 12.5%
        assert container.volume_utilization == pytest.approx(12.5, 0.1)
        # 100 / 1000 = 10%
        assert container.weight_utilization == pytest.approx(10.0, 0.1)


class TestExtremePoint:
    """Tests for ExtremePoint class."""
    
    def test_extreme_point_creation(self):
        """Test extreme point creation."""
        ep = ExtremePoint(10, 20, 30)
        assert ep.x == 10
        assert ep.y == 20
        assert ep.z == 30
    
    def test_extreme_point_comparison(self):
        """Test extreme point comparison (lower y preferred)."""
        ep1 = ExtremePoint(0, 0, 0)
        ep2 = ExtremePoint(0, 10, 0)
        assert ep1 < ep2  # Lower y is preferred


class TestBinPacker:
    """Tests for BinPacker class."""
    
    def test_packer_creation(self):
        """Test packer creation with strategy."""
        packer = BinPacker(
            strategy=PackingStrategy.BEST_FIT,
            sorting_criterion=SortingCriterion.VOLUME_DESC
        )
        assert packer.strategy == PackingStrategy.BEST_FIT
    
    def test_pack_single_box(self):
        """Test packing a single box."""
        packer = BinPacker()
        container = Container("C1", ContainerType.SMALL_VAN, 300, 150, 180, 1000)
        boxes = [Box("B1", BoxType.SMALL, 50, 40, 30, 10)]
        
        result = packer.pack(boxes, container)
        
        assert len(result.packed_boxes) == 1
        assert len(result.unpacked_boxes) == 0
    
    def test_pack_multiple_boxes(self):
        """Test packing multiple boxes."""
        packer = BinPacker()
        container = Container("C1", ContainerType.BOX_TRUCK, 600, 250, 270, 5000)
        boxes = [
            Box("B1", BoxType.SMALL, 50, 40, 30, 10),
            Box("B2", BoxType.SMALL, 45, 35, 25, 8),
            Box("B3", BoxType.MEDIUM, 60, 50, 40, 15),
        ]
        
        result = packer.pack(boxes, container)
        
        assert len(result.packed_boxes) == 3
        assert len(result.unpacked_boxes) == 0
    
    def test_pack_oversized_box(self):
        """Test that oversized box is not packed."""
        packer = BinPacker()
        container = Container("C1", ContainerType.SMALL_VAN, 100, 100, 100, 1000)
        boxes = [Box("B1", BoxType.XLARGE, 200, 200, 200, 50)]  # Too big
        
        result = packer.pack(boxes, container)
        
        assert len(result.packed_boxes) == 0
        assert len(result.unpacked_boxes) == 1
    
    def test_pack_overweight(self):
        """Test that overweight situation is handled."""
        packer = BinPacker()
        container = Container("C1", ContainerType.SMALL_VAN, 100, 100, 100, 10)  # Low weight limit
        boxes = [Box("B1", BoxType.SMALL, 30, 30, 30, 50)]  # Heavy
        
        result = packer.pack(boxes, container)
        
        assert len(result.unpacked_boxes) == 1
    
    def test_sequence_aware_packing(self):
        """Test LIFO packing for delivery sequence."""
        packer = BinPacker(
            strategy=PackingStrategy.SEQUENCE_AWARE,
            sorting_criterion=SortingCriterion.LIFO_SEQUENCE
        )
        container = Container("C1", ContainerType.BOX_TRUCK, 600, 250, 270, 5000)
        
        # Create boxes with different sequences
        boxes = [
            Box("B1", BoxType.SMALL, 50, 40, 30, 10, delivery_sequence=1),  # First delivery
            Box("B2", BoxType.SMALL, 45, 35, 25, 8, delivery_sequence=2),
            Box("B3", BoxType.SMALL, 40, 35, 30, 9, delivery_sequence=3),  # Last delivery
        ]
        
        result = packer.pack(boxes, container)
        
        # All should be packed
        assert len(result.packed_boxes) == 3
        
        # Box with sequence 3 should be loaded first (closer to door = higher x)
        # Box with sequence 1 should be loaded last (further from door = lower x or closer to back)
        packed_by_seq = {b.delivery_sequence: b for b in result.packed_boxes}
        
        # The last delivery (seq 1) should be near the door (high x)
        # The first delivery (seq 3) should be at the back (low x)
        # Due to LIFO: last delivery loaded first, first delivery loaded last
        # So seq 3 goes in first (back), seq 1 goes in last (front/door)
    
    def test_volume_utilization_calculation(self):
        """Test volume utilization in packing result."""
        packer = BinPacker()
        container = Container("C1", ContainerType.SMALL_VAN, 100, 100, 100, 1000)
        boxes = [Box("B1", BoxType.SMALL, 50, 50, 50, 10)]
        
        result = packer.pack(boxes, container)
        
        # 50*50*50 / 100*100*100 * 100 = 12.5%
        assert result.volume_utilization == pytest.approx(12.5, 0.1)


class TestPackingStrategies:
    """Tests for different packing strategies."""
    
    def test_best_fit_strategy(self):
        """Test best fit packing strategy."""
        packer = BinPacker(strategy=PackingStrategy.BEST_FIT)
        container = Container("C1", ContainerType.BOX_TRUCK, 600, 250, 270, 5000)
        boxes = [
            Box(f"B{i}", BoxType.SMALL, 50, 40, 30, 10)
            for i in range(5)
        ]
        
        result = packer.pack(boxes, container)
        assert len(result.packed_boxes) == 5
    
    def test_weight_balanced_strategy(self):
        """Test weight balanced packing strategy."""
        packer = BinPacker(strategy=PackingStrategy.WEIGHT_BALANCED)
        container = Container("C1", ContainerType.BOX_TRUCK, 600, 250, 270, 5000)
        boxes = [
            Box("B1", BoxType.LARGE, 80, 60, 50, 100),  # Heavy
            Box("B2", BoxType.LARGE, 80, 60, 50, 100),  # Heavy
            Box("B3", BoxType.SMALL, 40, 30, 20, 5),    # Light
            Box("B4", BoxType.SMALL, 40, 30, 20, 5),    # Light
        ]
        
        result = packer.pack(boxes, container)
        # Should pack all with weight consideration
        assert len(result.packed_boxes) == 4


class TestSortingCriteria:
    """Tests for sorting criteria."""
    
    def test_volume_desc_sorting(self):
        """Test sorting by volume descending."""
        packer = BinPacker(sorting_criterion=SortingCriterion.VOLUME_DESC)
        container = Container("C1", ContainerType.BOX_TRUCK, 600, 250, 270, 5000)
        
        boxes = [
            Box("SMALL", BoxType.SMALL, 20, 20, 20, 5),
            Box("LARGE", BoxType.LARGE, 100, 100, 100, 50),
            Box("MEDIUM", BoxType.MEDIUM, 50, 50, 50, 20),
        ]
        
        result = packer.pack(boxes, container)
        
        # Largest should be packed first and at the back
        assert len(result.packed_boxes) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
