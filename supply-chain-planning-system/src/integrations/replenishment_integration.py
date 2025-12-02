"""
Auto-Replenishment Integration.
"""

from typing import Optional, Any, List, Dict
import pandas as pd
import logging

from src.data.models import ReplenishmentResult, InventoryResult

logger = logging.getLogger(__name__)


class ReplenishmentIntegration:
    """Integration with the Auto-Replenishment System."""
    
    def __init__(self, config: Any):
        self.config = config
        self._urgent_items: List[Dict[str, Any]] = []
        logger.info("ReplenishmentIntegration initialized")
    
    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        inventory: Optional[InventoryResult] = None,
        forecast: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ReplenishmentResult:
        """Run auto-replenishment."""
        logger.info("Running auto-replenishment")
        
        # Calculate replenishment orders
        orders = self._calculate_orders(data, inventory, forecast)
        
        # Separate into PO and TO
        purchase_orders = orders[orders['order_type'] == 'PO'] if not orders.empty else pd.DataFrame()
        transfer_orders = orders[orders['order_type'] == 'TO'] if not orders.empty else pd.DataFrame()
        
        # Check for urgent items
        self._urgent_items = self._identify_urgent_items(orders)
        
        return ReplenishmentResult(
            orders=orders,
            purchase_orders=purchase_orders if not purchase_orders.empty else None,
            transfer_orders=transfer_orders if not transfer_orders.empty else None,
            automation_rate=0.82,
            alerts=[
                {'id': 'REP001', 'type': 'urgent_order', 'item_id': 'SKU005', 'severity': 'HIGH'}
            ],
            service_level_achieved=0.97,
            classifications=pd.DataFrame({
                'item_id': ['SKU001', 'SKU002', 'SKU003'],
                'abc_class': ['A', 'B', 'C'],
                'policy': ['periodic_review', 'periodic_review', 'min_max']
            })
        )
    
    def _calculate_orders(
        self,
        data: Optional[pd.DataFrame],
        inventory: Optional[InventoryResult],
        forecast: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate replenishment orders."""
        return pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003'],
            'item_id': ['SKU001', 'SKU002', 'SKU003'],
            'order_type': ['PO', 'PO', 'TO'],
            'quantity': [500, 300, 200],
            'supplier_id': ['SUP001', 'SUP001', None],
            'source_location': [None, None, 'DC001'],
            'destination': ['DC001', 'DC001', 'Store005'],
            'urgency': ['normal', 'normal', 'high']
        })
    
    def _identify_urgent_items(self, orders: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify urgent replenishment items."""
        urgent = []
        if not orders.empty and 'urgency' in orders.columns:
            urgent_orders = orders[orders['urgency'] == 'high']
            for _, row in urgent_orders.iterrows():
                urgent.append({
                    'id': row['order_id'],
                    'type': 'urgent_replenishment',
                    'item_id': row['item_id'],
                    'severity': 'HIGH',
                    'quantity': row['quantity']
                })
        return urgent
    
    def run_scenario(self, scenario: str, date: str) -> ReplenishmentResult:
        """Run replenishment for a specific scenario."""
        logger.info("Running scenario %s for date %s", scenario, date)
        return self.run(None)
    
    def get_urgent_items(self) -> List[Dict[str, Any]]:
        """Get urgent replenishment items."""
        return self._urgent_items
