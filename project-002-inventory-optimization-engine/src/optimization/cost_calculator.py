"""Inventory cost calculation module."""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Calculate various inventory-related costs.
    
    Includes holding costs, ordering costs, stockout costs, and total costs.
    """
    
    def __init__(
        self,
        holding_cost_rate: float = 0.25,
        ordering_cost: float = 100,
        stockout_cost_rate: float = 2.0
    ):
        """
        Initialize CostCalculator.
        
        Args:
            holding_cost_rate: Annual holding cost as % of unit cost (default 25%)
            ordering_cost: Fixed cost per order (default $100)
            stockout_cost_rate: Stockout cost as multiple of unit cost (default 2x)
        """
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.stockout_cost_rate = stockout_cost_rate
        
    def calculate_holding_cost(
        self,
        average_inventory: float,
        unit_cost: float,
        holding_period_days: int = 365
    ) -> float:
        """
        Calculate holding cost for inventory.
        
        Holding Cost = Average inventory * Unit cost * Holding rate * (Period / 365)
        
        Args:
            average_inventory: Average inventory level
            unit_cost: Cost per unit
            holding_period_days: Number of days in period
            
        Returns:
            Total holding cost for period
        """
        annual_rate = self.holding_cost_rate
        period_rate = annual_rate * (holding_period_days / 365)
        holding_cost = average_inventory * unit_cost * period_rate
        return max(0, holding_cost)
    
    def calculate_ordering_cost(
        self,
        num_orders: float,
        ordering_cost: float = None
    ) -> float:
        """
        Calculate total ordering cost.
        
        Ordering Cost = Number of orders * Fixed ordering cost
        
        Args:
            num_orders: Number of orders placed
            ordering_cost: Fixed cost per order (optional)
            
        Returns:
            Total ordering cost
        """
        cost_per_order = ordering_cost if ordering_cost is not None else self.ordering_cost
        return num_orders * cost_per_order
    
    def calculate_stockout_cost(
        self,
        stockout_quantity: float,
        unit_cost: float,
        stockout_cost_rate: float = None
    ) -> float:
        """
        Calculate cost of stockouts.
        
        Stockout Cost = Stockout quantity * Unit cost * Stockout rate
        
        Args:
            stockout_quantity: Quantity of unmet demand
            unit_cost: Cost per unit
            stockout_cost_rate: Stockout cost multiplier (optional)
            
        Returns:
            Total stockout cost
        """
        rate = stockout_cost_rate if stockout_cost_rate is not None else self.stockout_cost_rate
        stockout_cost = stockout_quantity * unit_cost * rate
        return max(0, stockout_cost)
    
    def calculate_purchase_cost(
        self,
        quantity: float,
        unit_cost: float
    ) -> float:
        """
        Calculate purchase cost.
        
        Args:
            quantity: Quantity purchased
            unit_cost: Cost per unit
            
        Returns:
            Total purchase cost
        """
        return quantity * unit_cost
    
    def calculate_total_inventory_cost(
        self,
        average_inventory: float,
        num_orders: float,
        stockout_quantity: float,
        purchase_quantity: float,
        unit_cost: float,
        holding_period_days: int = 365
    ) -> Dict[str, float]:
        """
        Calculate total inventory cost with breakdown.
        
        Args:
            average_inventory: Average inventory level
            num_orders: Number of orders
            stockout_quantity: Quantity of stockouts
            purchase_quantity: Total quantity purchased
            unit_cost: Cost per unit
            holding_period_days: Number of days in period
            
        Returns:
            Dictionary with cost breakdown
        """
        holding = self.calculate_holding_cost(average_inventory, unit_cost, holding_period_days)
        ordering = self.calculate_ordering_cost(num_orders)
        stockout = self.calculate_stockout_cost(stockout_quantity, unit_cost)
        purchase = self.calculate_purchase_cost(purchase_quantity, unit_cost)
        
        total = holding + ordering + stockout + purchase
        
        return {
            'holding_cost': holding,
            'ordering_cost': ordering,
            'stockout_cost': stockout,
            'purchase_cost': purchase,
            'total_cost': total,
            'operational_cost': holding + ordering + stockout,  # Excludes purchase
            'cost_breakdown': {
                'holding_pct': (holding / total * 100) if total > 0 else 0,
                'ordering_pct': (ordering / total * 100) if total > 0 else 0,
                'stockout_pct': (stockout / total * 100) if total > 0 else 0,
                'purchase_pct': (purchase / total * 100) if total > 0 else 0
            }
        }
    
    def calculate_cost_per_unit_sold(
        self,
        total_cost: float,
        units_sold: float
    ) -> float:
        """
        Calculate cost per unit sold.
        
        Args:
            total_cost: Total inventory cost
            units_sold: Number of units sold
            
        Returns:
            Cost per unit sold
        """
        if units_sold <= 0:
            return 0
        return total_cost / units_sold
    
    def calculate_inventory_turnover_cost(
        self,
        annual_sales: float,
        average_inventory: float,
        unit_cost: float
    ) -> Dict[str, float]:
        """
        Calculate inventory turnover metrics and associated costs.
        
        Args:
            annual_sales: Annual sales quantity
            average_inventory: Average inventory level
            unit_cost: Cost per unit
            
        Returns:
            Dictionary with turnover metrics and costs
        """
        if average_inventory <= 0:
            return {
                'turnover_ratio': 0,
                'days_of_supply': 0,
                'inventory_value': 0,
                'annual_holding_cost': 0
            }
        
        # Inventory turnover ratio
        turnover = annual_sales / average_inventory
        
        # Days of supply
        days_supply = 365 / turnover if turnover > 0 else 365
        
        # Average inventory value
        inventory_value = average_inventory * unit_cost
        
        # Annual holding cost
        annual_holding = inventory_value * self.holding_cost_rate
        
        return {
            'turnover_ratio': turnover,
            'days_of_supply': days_supply,
            'inventory_value': inventory_value,
            'annual_holding_cost': annual_holding
        }
    
    def calculate_service_level_cost_tradeoff(
        self,
        demand_mean: float,
        demand_std: float,
        unit_cost: float,
        service_levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate cost implications of different service levels.
        
        Args:
            demand_mean: Average demand
            demand_std: Standard deviation of demand
            unit_cost: Cost per unit
            service_levels: List of service levels to evaluate
            
        Returns:
            DataFrame with service level cost analysis
        """
        if service_levels is None:
            service_levels = [0.85, 0.90, 0.95, 0.99]
        
        from scipy import stats
        
        results = []
        
        for sl in service_levels:
            # Z-score for service level
            z = stats.norm.ppf(sl)
            
            # Safety stock required
            safety_stock = z * demand_std
            
            # Expected stockouts (simplified)
            expected_stockout = demand_std * stats.norm.pdf(z) * (1 - sl)
            
            # Costs
            holding_cost = self.calculate_holding_cost(safety_stock, unit_cost)
            stockout_cost = self.calculate_stockout_cost(expected_stockout, unit_cost)
            
            total_cost = holding_cost + stockout_cost
            
            results.append({
                'service_level': sl,
                'service_level_pct': sl * 100,
                'z_score': z,
                'safety_stock': safety_stock,
                'expected_stockout': expected_stockout,
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost,
                'total_cost': total_cost
            })
        
        return pd.DataFrame(results)
    
    def calculate_abc_class_costs(
        self,
        data: pd.DataFrame,
        group_col: str = 'abc_class',
        inventory_col: str = 'average_inventory',
        sales_col: str = 'sales_sum',
        price_col: str = 'sell_price_mean',
        orders_col: str = 'orders_per_year'
    ) -> pd.DataFrame:
        """
        Calculate costs by ABC class.
        
        Args:
            data: DataFrame with inventory data
            group_col: Column to group by (ABC class)
            inventory_col: Average inventory column
            sales_col: Sales column
            price_col: Unit price column
            orders_col: Number of orders column
            
        Returns:
            DataFrame with cost breakdown by class
        """
        logger.info(f"Calculating costs by {group_col}...")
        
        results = []
        
        for group_value in data[group_col].unique():
            group_data = data[data[group_col] == group_value]
            
            total_inventory = group_data[inventory_col].sum()
            total_sales = group_data[sales_col].sum()
            avg_price = group_data[price_col].mean()
            total_orders = group_data[orders_col].sum()
            
            # Calculate costs
            holding = self.calculate_holding_cost(total_inventory, avg_price)
            ordering = self.calculate_ordering_cost(total_orders)
            
            # Turnover metrics
            turnover_metrics = self.calculate_inventory_turnover_cost(
                total_sales, total_inventory, avg_price
            )
            
            results.append({
                group_col: group_value,
                'num_items': len(group_data),
                'total_inventory': total_inventory,
                'total_sales': total_sales,
                'inventory_value': total_inventory * avg_price,
                'holding_cost': holding,
                'ordering_cost': ordering,
                'total_operational_cost': holding + ordering,
                'turnover_ratio': turnover_metrics['turnover_ratio'],
                'days_of_supply': turnover_metrics['days_of_supply']
            })
        
        result_df = pd.DataFrame(results)
        
        # Calculate percentages
        result_df['cost_pct'] = (
            result_df['total_operational_cost'] /
            result_df['total_operational_cost'].sum() * 100
        ).round(2)
        
        logger.info(f"\nCost Summary by {group_col}:")
        logger.info(result_df.to_string())
        
        return result_df
