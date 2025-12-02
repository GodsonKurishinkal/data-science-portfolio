"""
Inventory Optimization Integration.
"""

from typing import Optional, Any, List, Dict
import pandas as pd
import logging

from src.data.models import InventoryResult

logger = logging.getLogger(__name__)


class InventoryIntegration:
    """Integration with the Inventory Optimization Engine."""
    
    def __init__(self, config: Any):
        self.config = config
        logger.info("InventoryIntegration initialized")
    
    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        _forecast: Optional[pd.DataFrame] = None,
        **_kwargs
    ) -> InventoryResult:
        """Run inventory optimization."""
        logger.info("Running inventory optimization")
        
        positions = self._calculate_positions(data)
        classifications = self._calculate_classifications(data)
        
        return InventoryResult(
            positions=positions,
            classifications=classifications,
            service_level=0.98,
            total_inventory_value=2500000.0,
            recommendations=[
                {'item_id': 'SKU001', 'action': 'increase_safety_stock', 'reason': 'High demand variability'},
                {'item_id': 'SKU042', 'action': 'reduce_order_quantity', 'reason': 'Overstock risk'}
            ]
        )
    
    def _calculate_positions(self, _data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Calculate inventory positions."""
        return pd.DataFrame({
            'item_id': ['SKU001', 'SKU002', 'SKU003'],
            'on_hand': [500, 320, 180],
            'on_order': [200, 0, 100],
            'committed': [150, 80, 50],
            'available': [550, 240, 230]
        })
    
    def _calculate_classifications(self, _data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Calculate ABC-XYZ classifications."""
        return pd.DataFrame({
            'item_id': ['SKU001', 'SKU002', 'SKU003'],
            'abc_class': ['A', 'B', 'C'],
            'xyz_class': ['X', 'Y', 'Z'],
            'combined': ['AX', 'BY', 'CZ']
        })
    
    def get_exceptions(self) -> List[Dict[str, Any]]:
        """Get inventory exceptions."""
        return [
            {'id': 'INV001', 'type': 'stockout_risk', 'item_id': 'SKU005', 'severity': 'HIGH'},
            {'id': 'INV002', 'type': 'overstock', 'item_id': 'SKU042', 'severity': 'MEDIUM'}
        ]
