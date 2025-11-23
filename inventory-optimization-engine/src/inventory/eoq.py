"""Economic Order Quantity (EOQ) calculation module."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EOQCalculator:
    """
    Calculate Economic Order Quantity and related metrics.
    
    EOQ minimizes total inventory costs (ordering + holding costs).
    """
    
    def __init__(
        self,
        ordering_cost: float = 100,
        holding_cost_rate: float = 0.25
    ):
        """
        Initialize EOQCalculator.
        
        Args:
            ordering_cost: Fixed cost per order
            holding_cost_rate: Annual holding cost as % of unit cost
        """
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
        
    def calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
        ordering_cost: float = None,
        holding_cost_rate: float = None
    ) -> float:
        """
        Calculate Economic Order Quantity.
        
        EOQ = √(2 * D * S / H)
        where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        
        Args:
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            ordering_cost: Fixed ordering cost (optional)
            holding_cost_rate: Annual holding cost rate (optional)
            
        Returns:
            Economic order quantity
        """
        S = ordering_cost if ordering_cost is not None else self.ordering_cost
        h = holding_cost_rate if holding_cost_rate is not None else self.holding_cost_rate
        H = unit_cost * h
        
        if annual_demand <= 0 or H <= 0:
            return 0
        
        eoq = np.sqrt((2 * annual_demand * S) / H)
        return max(1, eoq)
    
    def calculate_total_cost(
        self,
        order_quantity: float,
        annual_demand: float,
        unit_cost: float,
        ordering_cost: float = None,
        holding_cost_rate: float = None
    ) -> Dict[str, float]:
        """
        Calculate total inventory costs for a given order quantity.
        
        Total Cost = Ordering Cost + Holding Cost + Purchase Cost
        
        Args:
            order_quantity: Order quantity
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            ordering_cost: Fixed ordering cost (optional)
            holding_cost_rate: Annual holding cost rate (optional)
            
        Returns:
            Dictionary with cost breakdown
        """
        S = ordering_cost if ordering_cost is not None else self.ordering_cost
        h = holding_cost_rate if holding_cost_rate is not None else self.holding_cost_rate
        H = unit_cost * h
        
        # Number of orders per year
        num_orders = annual_demand / order_quantity if order_quantity > 0 else 0
        
        # Ordering cost
        annual_ordering_cost = num_orders * S
        
        # Holding cost (average inventory = Q/2)
        average_inventory = order_quantity / 2
        annual_holding_cost = average_inventory * H
        
        # Purchase cost
        annual_purchase_cost = annual_demand * unit_cost
        
        # Total cost
        total_cost = annual_ordering_cost + annual_holding_cost + annual_purchase_cost
        
        return {
            'order_quantity': order_quantity,
            'num_orders_per_year': num_orders,
            'ordering_cost': annual_ordering_cost,
            'holding_cost': annual_holding_cost,
            'purchase_cost': annual_purchase_cost,
            'total_cost': total_cost,
            'average_inventory': average_inventory
        }
    
    def calculate_eoq_with_backorders(
        self,
        annual_demand: float,
        unit_cost: float,
        stockout_cost: float,
        ordering_cost: float = None,
        holding_cost_rate: float = None
    ) -> Tuple[float, float]:
        """
        Calculate EOQ allowing for planned backorders.
        
        Args:
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            stockout_cost: Cost per unit backordered per year
            ordering_cost: Fixed ordering cost (optional)
            holding_cost_rate: Annual holding cost rate (optional)
            
        Returns:
            Tuple of (EOQ, maximum backorder level)
        """
        S = ordering_cost if ordering_cost is not None else self.ordering_cost
        h = holding_cost_rate if holding_cost_rate is not None else self.holding_cost_rate
        H = unit_cost * h
        
        # EOQ with backorders
        eoq_backorders = np.sqrt((2 * annual_demand * S) / H * (H + stockout_cost) / stockout_cost)
        
        # Maximum backorder level
        max_backorder = eoq_backorders * (H / (H + stockout_cost))
        
        return eoq_backorders, max_backorder
    
    def calculate_eoq_with_quantity_discount(
        self,
        annual_demand: float,
        discount_schedule: Dict[float, float],
        ordering_cost: float = None,
        holding_cost_rate: float = None
    ) -> Dict[str, float]:
        """
        Calculate optimal order quantity with quantity discounts.
        
        Args:
            annual_demand: Annual demand quantity
            discount_schedule: Dictionary of {min_quantity: unit_price}
            ordering_cost: Fixed ordering cost (optional)
            holding_cost_rate: Annual holding cost rate (optional)
            
        Returns:
            Dictionary with optimal order quantity and costs
        """
        S = ordering_cost if ordering_cost is not None else self.ordering_cost
        h = holding_cost_rate if holding_cost_rate is not None else self.holding_cost_rate
        
        best_option = None
        min_total_cost = float('inf')
        
        # Sort discount schedule by quantity
        sorted_schedule = sorted(discount_schedule.items())
        
        for min_qty, unit_price in sorted_schedule:
            # Calculate EOQ at this price point
            H = unit_price * h
            eoq = np.sqrt((2 * annual_demand * S) / H)
            
            # If EOQ is less than minimum quantity, use minimum quantity
            order_qty = max(eoq, min_qty)
            
            # Calculate total cost
            costs = self.calculate_total_cost(
                order_qty, annual_demand, unit_price, S, h
            )
            
            if costs['total_cost'] < min_total_cost:
                min_total_cost = costs['total_cost']
                best_option = {
                    'order_quantity': order_qty,
                    'unit_price': unit_price,
                    'min_quantity_threshold': min_qty,
                    **costs
                }
        
        return best_option
    
    def calculate_for_dataframe(
        self,
        data: pd.DataFrame,
        demand_col: str = 'sales_sum',
        price_col: str = 'sell_price_mean',
        annualization_factor: float = 365 / 28  # M5 has 28-day data
    ) -> pd.DataFrame:
        """
        Calculate EOQ for multiple items in a DataFrame.
        
        Args:
            data: DataFrame with demand and price data
            demand_col: Column name for demand
            price_col: Column name for unit price
            annualization_factor: Factor to convert demand to annual
            
        Returns:
            DataFrame with EOQ and related metrics
        """
        logger.info("Calculating EOQ for all items...")
        
        result = data.copy()
        
        # Annualize demand
        result['annual_demand'] = result[demand_col] * annualization_factor
        
        # Calculate EOQ
        result['eoq'] = result.apply(
            lambda row: self.calculate_eoq(
                row['annual_demand'],
                row[price_col]
            ),
            axis=1
        )
        
        # Calculate order frequency (orders per year)
        result['orders_per_year'] = result['annual_demand'] / result['eoq']
        
        # Calculate days between orders
        result['days_between_orders'] = 365 / result['orders_per_year']
        
        # Calculate average inventory
        result['average_inventory'] = result['eoq'] / 2
        
        # Calculate annual costs
        result['annual_ordering_cost'] = result['orders_per_year'] * self.ordering_cost
        result['annual_holding_cost'] = (
            result['average_inventory'] * result[price_col] * self.holding_cost_rate
        )
        result['annual_purchase_cost'] = result['annual_demand'] * result[price_col]
        result['total_annual_cost'] = (
            result['annual_ordering_cost'] +
            result['annual_holding_cost'] +
            result['annual_purchase_cost']
        )
        
        # Round EOQ to whole units
        result['eoq'] = result['eoq'].round(0).astype(int)
        
        logger.info(f"Average EOQ: {result['eoq'].mean():.2f}")
        logger.info(f"Average orders per year: {result['orders_per_year'].mean():.2f}")
        logger.info(f"Total annual cost: ${result['total_annual_cost'].sum():,.2f}")
        
        return result
    
    def calculate_reorder_frequency(
        self,
        eoq: float,
        demand_mean: float
    ) -> Dict[str, float]:
        """
        Calculate reorder frequency metrics.
        
        Args:
            eoq: Economic order quantity
            demand_mean: Average daily demand
            
        Returns:
            Dictionary with frequency metrics
        """
        if demand_mean <= 0 or eoq <= 0:
            return {
                'days_between_orders': 0,
                'orders_per_year': 0,
                'orders_per_month': 0
            }
        
        days_between_orders = eoq / demand_mean
        orders_per_year = 365 / days_between_orders
        orders_per_month = orders_per_year / 12
        
        return {
            'days_between_orders': days_between_orders,
            'orders_per_year': orders_per_year,
            'orders_per_month': orders_per_month
        }
    
    def optimize_order_quantity(
        self,
        annual_demand: float,
        unit_cost: float,
        min_order_qty: float = None,
        max_order_qty: float = None,
        ordering_cost: float = None,
        holding_cost_rate: float = None
    ) -> float:
        """
        Calculate optimal order quantity with constraints.
        
        Args:
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            min_order_qty: Minimum order quantity constraint
            max_order_qty: Maximum order quantity constraint
            ordering_cost: Fixed ordering cost (optional)
            holding_cost_rate: Annual holding cost rate (optional)
            
        Returns:
            Optimal order quantity
        """
        # Calculate unconstrained EOQ
        eoq = self.calculate_eoq(
            annual_demand, unit_cost, ordering_cost, holding_cost_rate
        )
        
        # Apply constraints
        if min_order_qty is not None:
            eoq = max(eoq, min_order_qty)
        
        if max_order_qty is not None:
            eoq = min(eoq, max_order_qty)
        
        return eoq
