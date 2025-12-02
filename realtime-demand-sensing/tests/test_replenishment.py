"""
Tests for Replenishment Engine module.

Tests the replenishment automation system including trigger
evaluation, order generation, and approval workflows.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.replenishment.engine import (
    ReplenishmentEngine,
    TriggerEvaluator,
    OrderGenerator,
    ApprovalWorkflow,
    InventoryPosition,
    ReplenishmentOrder,
    TriggerType,
    OrderPriority,
    OrderStatus
)


class TestInventoryPosition:
    """Tests for InventoryPosition dataclass."""
    
    def test_creation(self):
        """Test creating an InventoryPosition."""
        pos = InventoryPosition(
            product_id="TEST_001",
            on_hand=100,
            on_order=50,
            allocated=20,
            safety_stock=30,
            reorder_point=80,
            lead_time_days=7
        )
        
        assert pos.product_id == "TEST_001"
        assert pos.on_hand == 100
        assert pos.on_order == 50
        assert pos.allocated == 20
    
    def test_available_property(self):
        """Test available inventory calculation."""
        pos = InventoryPosition(
            product_id="TEST_001",
            on_hand=100,
            on_order=50,
            allocated=20,
            safety_stock=30,
            reorder_point=80,
            lead_time_days=7
        )
        
        # Available = on_hand + on_order - allocated = 100 + 50 - 20 = 130
        assert pos.available == 130
    
    def test_days_of_supply(self):
        """Test days of supply calculation."""
        pos = InventoryPosition(
            product_id="TEST_001",
            on_hand=100,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            avg_daily_demand=10,
            lead_time_days=7
        )
        
        # Days of supply = available / avg_daily_demand = 100 / 10 = 10
        assert pos.days_of_supply == 10.0
    
    def test_needs_replenishment(self):
        """Test needs_replenishment property."""
        pos_below = InventoryPosition(
            product_id="TEST_001",
            on_hand=50,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            lead_time_days=7
        )
        
        pos_above = InventoryPosition(
            product_id="TEST_002",
            on_hand=200,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            lead_time_days=7
        )
        
        assert pos_below.needs_replenishment is True
        assert pos_above.needs_replenishment is False


class TestReplenishmentOrder:
    """Tests for ReplenishmentOrder dataclass."""
    
    def test_creation(self):
        """Test creating a ReplenishmentOrder."""
        order = ReplenishmentOrder(
            id="ORD_001",
            product_id="TEST_001",
            quantity=100,
            trigger=TriggerType.REORDER_POINT,
            priority=OrderPriority.NORMAL,
            status=OrderStatus.PENDING_APPROVAL
        )
        
        assert order.id == "ORD_001"
        assert order.quantity == 100
        assert order.priority == OrderPriority.NORMAL
        assert order.status == OrderStatus.PENDING_APPROVAL
    
    def test_total_cost(self):
        """Test total cost calculation."""
        order = ReplenishmentOrder(
            id="ORD_001",
            product_id="TEST_001",
            quantity=100,
            trigger=TriggerType.REORDER_POINT,
            unit_cost=10.0
        )
        
        assert order.total_cost == 1000.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        order = ReplenishmentOrder(
            id="ORD_001",
            product_id="TEST_001",
            quantity=100,
            trigger=TriggerType.REORDER_POINT,
            unit_cost=10.0
        )
        
        d = order.to_dict()
        
        assert d['id'] == "ORD_001"
        assert d['product_id'] == "TEST_001"
        assert d['quantity'] == 100
        assert d['total_cost'] == 1000.0


class TestTriggerType:
    """Tests for TriggerType enum."""
    
    def test_trigger_types(self):
        """Test all trigger types exist."""
        assert TriggerType.STOCKOUT is not None
        assert TriggerType.SAFETY_STOCK is not None
        assert TriggerType.REORDER_POINT is not None
        assert TriggerType.FORECAST_BASED is not None
        assert TriggerType.MANUAL is not None
        assert TriggerType.SCHEDULED is not None


class TestOrderPriority:
    """Tests for OrderPriority enum."""
    
    def test_priority_levels(self):
        """Test all priority levels exist."""
        assert OrderPriority.EMERGENCY.value == 1
        assert OrderPriority.HIGH.value == 2
        assert OrderPriority.NORMAL.value == 3
        assert OrderPriority.LOW.value == 4


class TestTriggerEvaluator:
    """Tests for TriggerEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a TriggerEvaluator."""
        return TriggerEvaluator()
    
    @pytest.fixture
    def below_rop_position(self):
        """Create position below reorder point."""
        return InventoryPosition(
            product_id="TEST_001",
            on_hand=50,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            avg_daily_demand=10,
            lead_time_days=7
        )
    
    @pytest.fixture
    def stockout_position(self):
        """Create near-stockout position."""
        return InventoryPosition(
            product_id="TEST_001",
            on_hand=5,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            avg_daily_demand=10,
            lead_time_days=7
        )
    
    @pytest.fixture
    def healthy_position(self):
        """Create healthy inventory position."""
        return InventoryPosition(
            product_id="TEST_001",
            on_hand=200,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            avg_daily_demand=10,
            lead_time_days=7
        )
    
    def test_evaluate_stockout(self, evaluator, stockout_position):
        """Test stockout trigger detection."""
        triggers = evaluator.evaluate([stockout_position])
        
        # Should detect emergency
        assert len(triggers) > 0
        trigger_types = [t['trigger_type'] for t in triggers]
        assert TriggerType.STOCKOUT in trigger_types
    
    def test_evaluate_below_rop(self, evaluator, below_rop_position):
        """Test reorder point trigger detection."""
        triggers = evaluator.evaluate([below_rop_position])
        
        # Should detect below reorder point
        assert len(triggers) > 0
    
    def test_evaluate_healthy(self, evaluator, healthy_position):
        """Test no triggers for healthy position."""
        triggers = evaluator.evaluate([healthy_position])
        
        # Should have no triggers
        assert len(triggers) == 0
    
    def test_stockout_highest_priority(self, evaluator, stockout_position):
        """Test stockout has emergency priority."""
        triggers = evaluator.evaluate([stockout_position])
        
        stockout_triggers = [
            t for t in triggers
            if t['trigger_type'] == TriggerType.STOCKOUT
        ]
        
        if stockout_triggers:
            assert stockout_triggers[0]['priority'] == OrderPriority.EMERGENCY


class TestOrderGenerator:
    """Tests for OrderGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create an OrderGenerator."""
        return OrderGenerator(target_dos=30)
    
    @pytest.fixture
    def position(self):
        """Create inventory position."""
        return InventoryPosition(
            product_id="TEST_001",
            on_hand=50,
            on_order=0,
            allocated=0,
            safety_stock=30,
            reorder_point=80,
            avg_daily_demand=10,
            lead_time_days=7,
            unit_cost=10.0
        )
    
    @pytest.fixture
    def trigger(self, position):
        """Create a sample trigger."""
        return {
            'product_id': position.product_id,
            'trigger_type': TriggerType.REORDER_POINT,
            'priority': OrderPriority.NORMAL,
            'current_dos': position.days_of_supply,
            'position': position,
            'reason': 'Below reorder point'
        }
    
    def test_generate_orders(self, generator, trigger):
        """Test generating replenishment orders."""
        orders = generator.generate([trigger])
        
        assert len(orders) == 1
        order = orders[0]
        assert order.product_id == "TEST_001"
        assert order.quantity > 0
    
    def test_order_has_expected_delivery(self, generator, trigger):
        """Test orders have expected delivery dates."""
        orders = generator.generate([trigger])
        
        assert len(orders) == 1
        assert orders[0].expected_delivery is not None


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow class."""
    
    @pytest.fixture
    def workflow(self):
        """Create an ApprovalWorkflow."""
        return ApprovalWorkflow(
            auto_approve_limit=5000,
            auto_approve_emergency=True
        )
    
    @pytest.fixture
    def small_order(self):
        """Create a small order (below auto-approve limit)."""
        return ReplenishmentOrder(
            id="ORD_001",
            product_id="TEST_001",
            quantity=100,
            trigger=TriggerType.REORDER_POINT,
            priority=OrderPriority.NORMAL,
            unit_cost=10.0  # Total: 1000
        )
    
    @pytest.fixture
    def large_order(self):
        """Create a large order (above auto-approve limit)."""
        return ReplenishmentOrder(
            id="ORD_002",
            product_id="TEST_001",
            quantity=1000,
            trigger=TriggerType.FORECAST_BASED,
            priority=OrderPriority.NORMAL,
            unit_cost=10.0  # Total: 10,000
        )
    
    @pytest.fixture
    def emergency_order(self):
        """Create an emergency order."""
        return ReplenishmentOrder(
            id="ORD_003",
            product_id="TEST_001",
            quantity=500,
            trigger=TriggerType.STOCKOUT,
            priority=OrderPriority.EMERGENCY,
            unit_cost=10.0
        )
    
    def test_process_auto_approve_small(self, workflow, small_order):
        """Test auto-approval of small orders."""
        auto_approved, pending = workflow.process([small_order])
        
        assert len(auto_approved) == 1
        assert len(pending) == 0
        assert auto_approved[0].status == OrderStatus.APPROVED
    
    def test_process_pending_large(self, workflow, large_order):
        """Test large orders require approval."""
        auto_approved, pending = workflow.process([large_order])
        
        assert len(auto_approved) == 0
        assert len(pending) == 1
    
    def test_emergency_auto_approve(self, workflow, emergency_order):
        """Test emergency orders are auto-approved."""
        auto_approved, pending = workflow.process([emergency_order])
        
        assert len(auto_approved) == 1
        assert auto_approved[0].status == OrderStatus.APPROVED
    
    def test_manual_approval(self, workflow, large_order):
        """Test manual order approval."""
        result = workflow.approve(large_order, "manager1", "Approved for Q4")
        
        assert result is True
        assert large_order.status == OrderStatus.APPROVED
        assert large_order.approved_by == "manager1"
    
    def test_rejection(self, workflow, large_order):
        """Test order rejection."""
        workflow.reject(large_order, "Budget constraints")
        
        assert large_order.status == OrderStatus.REJECTED


class TestReplenishmentEngine:
    """Tests for ReplenishmentEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a ReplenishmentEngine."""
        return ReplenishmentEngine(
            target_dos=14,
            auto_approve_limit=5000,
            auto_approve_emergency=True
        )
    
    @pytest.fixture
    def sample_inventory_df(self):
        """Create sample inventory DataFrame."""
        return pd.DataFrame({
            'product_id': ['TEST_001', 'TEST_002', 'TEST_003'],
            'on_hand': [50, 200, 10],
            'on_order': [0, 50, 0],
            'allocated': [10, 20, 0],
            'safety_stock': [30, 30, 30],
            'reorder_point': [80, 80, 80],
            'avg_daily_demand': [10, 10, 10],
            'lead_time_days': [7, 7, 7],
            'unit_cost': [10.0, 10.0, 10.0]
        })
    
    def test_initialization(self):
        """Test engine initializes correctly."""
        engine = ReplenishmentEngine()
        assert engine is not None
    
    def test_update_positions(self, engine, sample_inventory_df):
        """Test updating inventory positions."""
        engine.update_positions(sample_inventory_df)
        
        assert len(engine.positions) == 3
        assert 'TEST_001' in engine.positions
    
    def test_run_cycle(self, engine, sample_inventory_df):
        """Test running a replenishment cycle."""
        engine.update_positions(sample_inventory_df)
        result = engine.run_cycle()
        
        assert 'triggers' in result
        assert 'orders_generated' in result
        assert 'auto_approved' in result
        assert 'pending_approval' in result
    
    def test_run_cycle_generates_orders(self, engine, sample_inventory_df):
        """Test cycle generates orders for low inventory."""
        engine.update_positions(sample_inventory_df)
        result = engine.run_cycle()
        
        # TEST_001 and TEST_003 are below ROP
        assert result['orders_generated'] > 0
    
    def test_get_pending_orders(self, engine, sample_inventory_df):
        """Test getting pending orders."""
        engine.update_positions(sample_inventory_df)
        engine.run_cycle()
        
        pending = engine.get_pending_orders()
        assert isinstance(pending, list)
    
    def test_get_active_orders(self, engine, sample_inventory_df):
        """Test getting active orders."""
        engine.update_positions(sample_inventory_df)
        engine.run_cycle()
        
        active = engine.get_active_orders()
        assert isinstance(active, list)
    
    def test_approve_order(self, engine, sample_inventory_df):
        """Test approving an order."""
        engine.update_positions(sample_inventory_df)
        engine.run_cycle()
        
        pending = engine.get_pending_orders()
        if pending:
            result = engine.approve_order(pending[0].id, "manager1")
            assert result is True
    
    def test_get_summary(self, engine, sample_inventory_df):
        """Test getting summary."""
        engine.update_positions(sample_inventory_df)
        engine.run_cycle()
        
        summary = engine.get_summary()
        
        assert 'total_positions' in summary
        assert 'pending_orders' in summary
        assert 'active_orders' in summary
    
    def test_orders_to_dataframe(self, engine, sample_inventory_df):
        """Test converting orders to DataFrame."""
        engine.update_positions(sample_inventory_df)
        engine.run_cycle()
        
        df = engine.orders_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)


class TestReplenishmentEdgeCases:
    """Test edge cases for replenishment system."""
    
    def test_zero_demand(self):
        """Test handling of zero demand."""
        engine = ReplenishmentEngine()
        
        inventory_df = pd.DataFrame({
            'product_id': ['TEST_001'],
            'on_hand': [100],
            'on_order': [0],
            'allocated': [0],
            'safety_stock': [30],
            'reorder_point': [80],
            'avg_daily_demand': [0],  # Zero demand
            'lead_time_days': [7],
            'unit_cost': [10.0]
        })
        
        engine.update_positions(inventory_df)
        result = engine.run_cycle()
        
        # Should handle gracefully
        assert result is not None
    
    def test_very_high_demand(self):
        """Test handling of very high demand."""
        engine = ReplenishmentEngine()
        
        inventory_df = pd.DataFrame({
            'product_id': ['TEST_001'],
            'on_hand': [10],
            'on_order': [0],
            'allocated': [0],
            'safety_stock': [100],
            'reorder_point': [200],
            'avg_daily_demand': [100],  # Very high
            'lead_time_days': [7],
            'unit_cost': [10.0]
        })
        
        engine.update_positions(inventory_df)
        result = engine.run_cycle()
        
        # Should generate order
        assert result['orders_generated'] > 0
    
    def test_empty_inventory(self):
        """Test handling of empty inventory DataFrame."""
        engine = ReplenishmentEngine()
        
        empty_df = pd.DataFrame(columns=[
            'product_id', 'on_hand', 'on_order', 'allocated',
            'safety_stock', 'reorder_point', 'avg_daily_demand',
            'lead_time_days', 'unit_cost'
        ])
        
        engine.update_positions(empty_df)
        result = engine.run_cycle()
        
        assert result['triggers'] == 0
        assert result['orders_generated'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
