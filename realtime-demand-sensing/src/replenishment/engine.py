"""
Replenishment Engine - Automated Inventory Replenishment

Automated replenishment system:
- Trigger evaluation based on inventory levels
- Order generation with safety stock
- Priority-based execution
- Human-in-the-loop approval workflows

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of replenishment triggers."""
    REORDER_POINT = "reorder_point"
    SAFETY_STOCK = "safety_stock"
    STOCKOUT = "stockout"
    FORECAST_BASED = "forecast_based"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class OrderPriority(Enum):
    """Order priority levels."""
    EMERGENCY = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class OrderStatus(Enum):
    """Order status states."""
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class InventoryPosition:
    """Current inventory position for a product."""
    product_id: str
    on_hand: float
    on_order: float = 0.0
    allocated: float = 0.0
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    avg_daily_demand: float = 0.0
    lead_time_days: float = 3.0
    unit_cost: float = 0.0
    
    @property
    def available(self) -> float:
        """Available inventory (on-hand + on-order - allocated)."""
        return self.on_hand + self.on_order - self.allocated
    
    @property
    def days_of_supply(self) -> float:
        """Days of supply based on average demand."""
        if self.avg_daily_demand <= 0:
            return float('inf')
        return self.available / self.avg_daily_demand
    
    @property
    def needs_replenishment(self) -> bool:
        """Check if replenishment is needed."""
        return self.available <= self.reorder_point


@dataclass
class ReplenishmentOrder:
    """Replenishment order details."""
    id: str
    product_id: str
    quantity: float
    trigger: TriggerType
    priority: OrderPriority = OrderPriority.NORMAL
    status: OrderStatus = OrderStatus.PENDING_APPROVAL
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    expected_delivery: Optional[datetime] = None
    unit_cost: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        """Total order cost."""
        return self.quantity * self.unit_cost
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'quantity': self.quantity,
            'trigger': self.trigger.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at,
            'approved_at': self.approved_at,
            'approved_by': self.approved_by,
            'expected_delivery': self.expected_delivery,
            'unit_cost': self.unit_cost,
            'total_cost': self.total_cost,
            'notes': self.notes
        }


class TriggerEvaluator:
    """
    Evaluates replenishment triggers.
    
    Monitors inventory positions and generates triggers
    when replenishment is needed.
    
    Example:
        >>> evaluator = TriggerEvaluator()
        >>> triggers = evaluator.evaluate(positions)
    """
    
    def __init__(
        self,
        stockout_threshold_days: float = 1.0,
        emergency_threshold_days: float = 0.5,
        forecast_horizon_days: int = 7
    ):
        """
        Initialize trigger evaluator.
        
        Args:
            stockout_threshold_days: Days of supply for stockout alert
            emergency_threshold_days: Days of supply for emergency
            forecast_horizon_days: Horizon for forecast-based triggers
        """
        self.stockout_threshold_days = stockout_threshold_days
        self.emergency_threshold_days = emergency_threshold_days
        self.forecast_horizon_days = forecast_horizon_days
        
        logger.info(
            "Initialized TriggerEvaluator: stockout=%.1f days, emergency=%.1f days",
            stockout_threshold_days, emergency_threshold_days
        )
    
    def evaluate(
        self,
        positions: List[InventoryPosition],
        forecasts: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Evaluate triggers for all positions.
        
        Args:
            positions: List of inventory positions
            forecasts: Optional forecasted demand by product_id
        
        Returns:
            List of triggered items with trigger info
        """
        triggers = []
        
        for pos in positions:
            trigger_info = self._evaluate_position(pos, forecasts)
            if trigger_info:
                triggers.append(trigger_info)
        
        # Sort by priority
        triggers.sort(key=lambda x: x['priority'].value)
        
        return triggers
    
    def _evaluate_position(
        self,
        pos: InventoryPosition,
        forecasts: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """Evaluate single position."""
        
        # Check for stockout (emergency)
        if pos.days_of_supply <= self.emergency_threshold_days:
            return {
                'product_id': pos.product_id,
                'trigger_type': TriggerType.STOCKOUT,
                'priority': OrderPriority.EMERGENCY,
                'current_dos': pos.days_of_supply,
                'position': pos,
                'reason': f"Stockout imminent: {pos.days_of_supply:.1f} days of supply"
            }
        
        # Check safety stock breach
        if pos.available < pos.safety_stock:
            return {
                'product_id': pos.product_id,
                'trigger_type': TriggerType.SAFETY_STOCK,
                'priority': OrderPriority.HIGH,
                'current_dos': pos.days_of_supply,
                'position': pos,
                'reason': f"Below safety stock: {pos.available:.0f} < {pos.safety_stock:.0f}"
            }
        
        # Check reorder point
        if pos.needs_replenishment:
            return {
                'product_id': pos.product_id,
                'trigger_type': TriggerType.REORDER_POINT,
                'priority': OrderPriority.NORMAL,
                'current_dos': pos.days_of_supply,
                'position': pos,
                'reason': f"At reorder point: {pos.available:.0f} <= {pos.reorder_point:.0f}"
            }
        
        # Check forecast-based trigger
        if forecasts and pos.product_id in forecasts:
            forecast_demand = forecasts[pos.product_id]
            if pos.available < forecast_demand * 1.2:  # 20% buffer
                return {
                    'product_id': pos.product_id,
                    'trigger_type': TriggerType.FORECAST_BASED,
                    'priority': OrderPriority.NORMAL,
                    'current_dos': pos.days_of_supply,
                    'position': pos,
                    'reason': f"Forecast-based: demand={forecast_demand:.0f}, available={pos.available:.0f}"
                }
        
        return None


class OrderGenerator:
    """
    Generates replenishment orders.
    
    Calculates order quantities based on:
    - EOQ (Economic Order Quantity)
    - Days of supply targets
    - Minimum/maximum order constraints
    
    Example:
        >>> generator = OrderGenerator(target_dos=14)
        >>> orders = generator.generate(triggers)
    """
    
    def __init__(
        self,
        target_dos: float = 14.0,
        min_order_value: float = 100.0,
        use_eoq: bool = True,
        ordering_cost: float = 50.0,
        holding_cost_pct: float = 0.25
    ):
        """
        Initialize order generator.
        
        Args:
            target_dos: Target days of supply
            min_order_value: Minimum order value
            use_eoq: Whether to use EOQ calculation
            ordering_cost: Fixed cost per order
            holding_cost_pct: Annual holding cost as % of unit cost
        """
        self.target_dos = target_dos
        self.min_order_value = min_order_value
        self.use_eoq = use_eoq
        self.ordering_cost = ordering_cost
        self.holding_cost_pct = holding_cost_pct
        
        self._order_counter = 0
        
        logger.info(
            "Initialized OrderGenerator: target_dos=%.1f, min_value=%.2f",
            target_dos, min_order_value
        )
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PO_{datetime.now().strftime('%Y%m%d')}_{self._order_counter:04d}"
    
    def _calculate_eoq(self, pos: InventoryPosition) -> float:
        """Calculate Economic Order Quantity."""
        annual_demand = pos.avg_daily_demand * 365
        holding_cost = pos.unit_cost * self.holding_cost_pct
        
        if holding_cost <= 0 or annual_demand <= 0:
            return pos.avg_daily_demand * self.target_dos
        
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / holding_cost)
        return eoq
    
    def _calculate_quantity(
        self,
        pos: InventoryPosition,
        trigger_type: TriggerType
    ) -> float:
        """Calculate order quantity."""
        
        # Target inventory level
        target_level = pos.avg_daily_demand * self.target_dos + pos.safety_stock
        
        # Current position gap
        gap = target_level - pos.available
        
        if self.use_eoq:
            eoq = self._calculate_eoq(pos)
            # Use EOQ or gap, whichever is larger
            quantity = max(eoq, gap)
        else:
            quantity = gap
        
        # Emergency orders: add extra buffer
        if trigger_type == TriggerType.STOCKOUT:
            quantity = max(quantity, pos.avg_daily_demand * 7)
        
        # Ensure positive
        quantity = max(0, quantity)
        
        return quantity
    
    def generate(
        self,
        triggers: List[Dict]
    ) -> List[ReplenishmentOrder]:
        """
        Generate orders from triggers.
        
        Args:
            triggers: List of triggered items from TriggerEvaluator
        
        Returns:
            List of ReplenishmentOrder objects
        """
        orders = []
        
        for trigger in triggers:
            pos = trigger['position']
            trigger_type = trigger['trigger_type']
            priority = trigger['priority']
            
            quantity = self._calculate_quantity(pos, trigger_type)
            
            # Skip if below minimum value
            if quantity * pos.unit_cost < self.min_order_value:
                logger.debug(
                    "Skipping %s: order value %.2f below minimum",
                    pos.product_id, quantity * pos.unit_cost
                )
                continue
            
            order = ReplenishmentOrder(
                id=self._generate_order_id(),
                product_id=pos.product_id,
                quantity=quantity,
                trigger=trigger_type,
                priority=priority,
                unit_cost=pos.unit_cost,
                expected_delivery=datetime.now() + timedelta(days=pos.lead_time_days)
            )
            
            order.notes.append(trigger['reason'])
            
            orders.append(order)
            
            logger.info(
                "Generated order %s: %s, qty=%.0f, priority=%s",
                order.id, pos.product_id, quantity, priority.name
            )
        
        return orders


class ApprovalWorkflow:
    """
    Manages order approval workflow.
    
    Features:
    - Auto-approval based on rules
    - Priority-based routing
    - Approval limits by user
    
    Example:
        >>> workflow = ApprovalWorkflow(auto_approve_limit=1000)
        >>> approved, pending = workflow.process(orders)
    """
    
    def __init__(
        self,
        auto_approve_limit: float = 1000.0,
        auto_approve_emergency: bool = True,
        approval_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize approval workflow.
        
        Args:
            auto_approve_limit: Auto-approve orders below this value
            auto_approve_emergency: Auto-approve emergency orders
            approval_limits: Approval limits by user
        """
        self.auto_approve_limit = auto_approve_limit
        self.auto_approve_emergency = auto_approve_emergency
        self.approval_limits = approval_limits or {}
        
        logger.info(
            "Initialized ApprovalWorkflow: auto_limit=%.2f, emergency_auto=%s",
            auto_approve_limit, auto_approve_emergency
        )
    
    def process(
        self,
        orders: List[ReplenishmentOrder]
    ) -> tuple:
        """
        Process orders through approval workflow.
        
        Args:
            orders: List of orders to process
        
        Returns:
            Tuple of (auto_approved, pending_approval) orders
        """
        auto_approved = []
        pending = []
        
        for order in orders:
            if self._should_auto_approve(order):
                order.status = OrderStatus.APPROVED
                order.approved_at = datetime.now()
                order.approved_by = "auto_approval"
                order.notes.append("Auto-approved by system")
                auto_approved.append(order)
                
                logger.info("Auto-approved order %s (%.2f)", order.id, order.total_cost)
            else:
                pending.append(order)
                logger.info("Order %s pending approval (%.2f)", order.id, order.total_cost)
        
        return auto_approved, pending
    
    def _should_auto_approve(self, order: ReplenishmentOrder) -> bool:
        """Check if order should be auto-approved."""
        
        # Emergency orders
        if self.auto_approve_emergency and order.priority == OrderPriority.EMERGENCY:
            return True
        
        # Below auto-approve limit
        if order.total_cost <= self.auto_approve_limit:
            return True
        
        return False
    
    def approve(
        self,
        order: ReplenishmentOrder,
        approver: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually approve an order.
        
        Args:
            order: Order to approve
            approver: Approver username
            notes: Approval notes
        
        Returns:
            True if approved successfully
        """
        # Check approval limit
        if approver in self.approval_limits:
            if order.total_cost > self.approval_limits[approver]:
                logger.warning(
                    "Approver %s exceeded limit for order %s",
                    approver, order.id
                )
                return False
        
        order.status = OrderStatus.APPROVED
        order.approved_at = datetime.now()
        order.approved_by = approver
        
        if notes:
            order.notes.append(f"Approved: {notes}")
        
        logger.info("Order %s approved by %s", order.id, approver)
        return True
    
    def reject(
        self,
        order: ReplenishmentOrder,
        reason: str
    ) -> None:
        """Reject an order."""
        order.status = OrderStatus.REJECTED
        order.notes.append(f"Rejected: {reason}")
        logger.info("Order %s rejected: %s", order.id, reason)


class ReplenishmentEngine:
    """
    Main replenishment automation engine.
    
    Orchestrates the full replenishment workflow:
    1. Evaluate triggers from inventory positions
    2. Generate orders based on triggers
    3. Process through approval workflow
    4. Track order execution
    
    Example:
        >>> engine = ReplenishmentEngine()
        >>> engine.update_positions(inventory_df)
        >>> results = engine.run_cycle()
    """
    
    def __init__(
        self,
        target_dos: float = 14.0,
        auto_approve_limit: float = 1000.0,
        auto_approve_emergency: bool = True
    ):
        """
        Initialize replenishment engine.
        
        Args:
            target_dos: Target days of supply
            auto_approve_limit: Auto-approve limit
            auto_approve_emergency: Auto-approve emergencies
        """
        self.trigger_evaluator = TriggerEvaluator()
        self.order_generator = OrderGenerator(target_dos=target_dos)
        self.approval_workflow = ApprovalWorkflow(
            auto_approve_limit=auto_approve_limit,
            auto_approve_emergency=auto_approve_emergency
        )
        
        self.positions: Dict[str, InventoryPosition] = {}
        self.orders: Dict[str, ReplenishmentOrder] = {}
        self.order_history: deque = deque(maxlen=1000)
        
        logger.info("Initialized ReplenishmentEngine")
    
    def update_positions(self, inventory_data: pd.DataFrame) -> None:
        """
        Update inventory positions from DataFrame.
        
        Expected columns:
        - product_id, on_hand, on_order, allocated
        - safety_stock, reorder_point, avg_daily_demand
        - lead_time_days, unit_cost
        """
        for _, row in inventory_data.iterrows():
            pos = InventoryPosition(
                product_id=row['product_id'],
                on_hand=row.get('on_hand', 0),
                on_order=row.get('on_order', 0),
                allocated=row.get('allocated', 0),
                safety_stock=row.get('safety_stock', 0),
                reorder_point=row.get('reorder_point', 0),
                avg_daily_demand=row.get('avg_daily_demand', 0),
                lead_time_days=row.get('lead_time_days', 3),
                unit_cost=row.get('unit_cost', 0)
            )
            self.positions[pos.product_id] = pos
        
        logger.info("Updated %d inventory positions", len(self.positions))
    
    def run_cycle(
        self,
        forecasts: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Run a replenishment cycle.
        
        Args:
            forecasts: Optional demand forecasts by product
        
        Returns:
            Cycle results summary
        """
        # Evaluate triggers
        triggers = self.trigger_evaluator.evaluate(
            list(self.positions.values()),
            forecasts
        )
        
        logger.info("Evaluated %d triggers", len(triggers))
        
        if not triggers:
            return {
                'triggers': 0,
                'orders_generated': 0,
                'auto_approved': 0,
                'pending_approval': 0,
                'total_value': 0
            }
        
        # Generate orders
        orders = self.order_generator.generate(triggers)
        
        # Process approvals
        auto_approved, pending = self.approval_workflow.process(orders)
        
        # Store orders
        for order in orders:
            self.orders[order.id] = order
        
        # Calculate totals
        total_value = sum(o.total_cost for o in orders)
        approved_value = sum(o.total_cost for o in auto_approved)
        
        result = {
            'triggers': len(triggers),
            'orders_generated': len(orders),
            'auto_approved': len(auto_approved),
            'pending_approval': len(pending),
            'total_value': total_value,
            'approved_value': approved_value,
            'orders': orders
        }
        
        logger.info(
            "Cycle complete: %d triggers, %d orders ($%.2f), %d auto-approved",
            len(triggers), len(orders), total_value, len(auto_approved)
        )
        
        return result
    
    def get_pending_orders(self) -> List[ReplenishmentOrder]:
        """Get orders pending approval."""
        return [o for o in self.orders.values() 
                if o.status == OrderStatus.PENDING_APPROVAL]
    
    def get_active_orders(self) -> List[ReplenishmentOrder]:
        """Get active orders (approved or in transit)."""
        active_statuses = {OrderStatus.APPROVED, OrderStatus.SUBMITTED, OrderStatus.IN_TRANSIT}
        return [o for o in self.orders.values() if o.status in active_statuses]
    
    def approve_order(
        self,
        order_id: str,
        approver: str,
        notes: Optional[str] = None
    ) -> bool:
        """Approve a pending order."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        return self.approval_workflow.approve(order, approver, notes)
    
    def reject_order(self, order_id: str, reason: str) -> bool:
        """Reject a pending order."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        self.approval_workflow.reject(order, reason)
        return True
    
    def get_summary(self) -> Dict:
        """Get replenishment summary."""
        pending = self.get_pending_orders()
        active = self.get_active_orders()
        
        return {
            'total_positions': len(self.positions),
            'positions_below_rop': sum(1 for p in self.positions.values() if p.needs_replenishment),
            'pending_orders': len(pending),
            'pending_value': sum(o.total_cost for o in pending),
            'active_orders': len(active),
            'active_value': sum(o.total_cost for o in active),
            'emergency_orders': sum(1 for o in self.orders.values() 
                                   if o.priority == OrderPriority.EMERGENCY)
        }
    
    def orders_to_dataframe(self) -> pd.DataFrame:
        """Convert orders to DataFrame."""
        if not self.orders:
            return pd.DataFrame()
        
        return pd.DataFrame([o.to_dict() for o in self.orders.values()])


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Replenishment Engine Test")
    print("=" * 60)
    
    # Create sample inventory data
    np.random.seed(42)
    n_products = 10
    
    inventory_df = pd.DataFrame({
        'product_id': [f"PROD_{i:03d}" for i in range(n_products)],
        'on_hand': np.random.uniform(10, 500, n_products),
        'on_order': np.random.choice([0, 50, 100], n_products),
        'allocated': np.random.uniform(0, 50, n_products),
        'safety_stock': np.random.uniform(20, 100, n_products),
        'reorder_point': np.random.uniform(50, 200, n_products),
        'avg_daily_demand': np.random.uniform(5, 50, n_products),
        'lead_time_days': np.random.choice([3, 5, 7], n_products),
        'unit_cost': np.random.uniform(5, 50, n_products)
    })
    
    # Force some low inventory situations
    inventory_df.loc[0, 'on_hand'] = 5  # Very low
    inventory_df.loc[1, 'on_hand'] = 30  # Below ROP
    
    print("\nSample Inventory Positions:")
    print(inventory_df[['product_id', 'on_hand', 'reorder_point', 'avg_daily_demand']].head())
    
    # Initialize engine
    engine = ReplenishmentEngine(
        target_dos=14,
        auto_approve_limit=500,
        auto_approve_emergency=True
    )
    
    # Update positions
    engine.update_positions(inventory_df)
    
    # Run replenishment cycle
    print("\n" + "-" * 40)
    results = engine.run_cycle()
    
    print(f"\nResults:")
    print(f"  Triggers evaluated: {results['triggers']}")
    print(f"  Orders generated: {results['orders_generated']}")
    print(f"  Auto-approved: {results['auto_approved']}")
    print(f"  Pending approval: {results['pending_approval']}")
    print(f"  Total value: ${results['total_value']:,.2f}")
    
    # Show orders
    print("\nGenerated Orders:")
    orders_df = engine.orders_to_dataframe()
    if not orders_df.empty:
        print(orders_df[['id', 'product_id', 'quantity', 'priority', 'status', 'total_cost']].to_string())
    
    # Get summary
    print("\nSummary:")
    summary = engine.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nReplenishment Engine test complete!")
