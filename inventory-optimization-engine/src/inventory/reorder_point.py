"""Reorder point calculation module."""

import pandas as pd
from typing import Union, Dict
import logging

logger = logging.getLogger(__name__)


class ReorderPointCalculator:
    """
    Calculate reorder points for inventory management.

    Reorder Point (ROP) = Average demand during lead time + Safety stock
    """

    def __init__(self, lead_time: int = 7):
        """
        Initialize ReorderPointCalculator.

        Args:
            lead_time: Lead time in days
        """
        self.lead_time = lead_time

    def calculate_reorder_point(
        self,
        demand_mean: float,
        safety_stock: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate basic reorder point.

        ROP = (Average daily demand * Lead time) + Safety stock

        Args:
            demand_mean: Average daily demand
            safety_stock: Safety stock quantity
            lead_time: Lead time in days (optional)

        Returns:
            Reorder point quantity
        """
        lt = lead_time if lead_time is not None else self.lead_time
        rop = (demand_mean * lt) + safety_stock
        return max(0, rop)

    def calculate_dynamic_reorder_point(
        self,
        demand_forecast: float,
        _demand_std: float,
        safety_stock: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate dynamic reorder point using forecasted demand.

        Args:
            demand_forecast: Forecasted daily demand
            demand_std: Standard deviation of demand
            safety_stock: Safety stock quantity
            lead_time: Lead time in days

        Returns:
            Dynamic reorder point
        """
        lt = lead_time if lead_time is not None else self.lead_time
        expected_demand = demand_forecast * lt
        rop = expected_demand + safety_stock
        return max(0, rop)

    def calculate_for_dataframe(
        self,
        data: pd.DataFrame,
        demand_mean_col: str = 'sales_mean',
        safety_stock_col: str = 'safety_stock',
        lead_time_col: str = None
    ) -> pd.DataFrame:
        """
        Calculate reorder points for multiple items.

        Args:
            data: DataFrame with demand and safety stock data
            demand_mean_col: Column name for mean demand
            safety_stock_col: Column name for safety stock
            lead_time_col: Column name for lead time (optional)

        Returns:
            DataFrame with reorder point column added
        """
        logger.info("Calculating reorder points...")

        result = data.copy()

        if lead_time_col and lead_time_col in result.columns:
            result['reorder_point'] = result.apply(
                lambda row: self.calculate_reorder_point(
                    row[demand_mean_col],
                    row[safety_stock_col],
                    row[lead_time_col]
                ),
                axis=1
            )
        else:
            result['reorder_point'] = result.apply(
                lambda row: self.calculate_reorder_point(
                    row[demand_mean_col],
                    row[safety_stock_col]
                ),
                axis=1
            )

        # Round to whole units
        result['reorder_point'] = result['reorder_point'].round(0).astype(int)

        logger.info("Average reorder point: %.2f", result['reorder_point'].mean())
        logger.info("Total reorder point: %.0f", result['reorder_point'].sum())

        return result

    def calculate_max_inventory_level(
        self,
        demand_mean: float,
        safety_stock: float,
        order_quantity: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate maximum inventory level for periodic review system.

        Max Level = (Demand * (Lead time + Review period)) + Safety stock

        Args:
            demand_mean: Average daily demand
            safety_stock: Safety stock quantity
            order_quantity: Order quantity (used as proxy for review period)
            lead_time: Lead time in days

        Returns:
            Maximum inventory level
        """
        lt = lead_time if lead_time is not None else self.lead_time
        review_period = order_quantity / demand_mean if demand_mean > 0 else 7
        max_level = (demand_mean * (lt + review_period)) + safety_stock
        return max(0, max_level)

    def determine_order_quantity_trigger(
        self,
        current_inventory: float,
        reorder_point: float,
        order_quantity: float
    ) -> Dict[str, Union[bool, float]]:
        """
        Determine if an order should be placed and how much.

        Args:
            current_inventory: Current inventory level
            reorder_point: Reorder point
            order_quantity: Standard order quantity

        Returns:
            Dictionary with order decision and quantity
        """
        should_order = current_inventory <= reorder_point

        if should_order:
            # Calculate order quantity to reach target level
            shortage = reorder_point - current_inventory + order_quantity
            order_qty = max(order_quantity, shortage)
        else:
            order_qty = 0

        return {
            'should_order': should_order,
            'order_quantity': order_qty,
            'inventory_position': current_inventory,
            'days_until_stockout': (current_inventory / order_quantity) if order_quantity > 0 else 0
        }

    def calculate_time_to_reorder(
        self,
        current_inventory: float,
        reorder_point: float,
        demand_mean: float
    ) -> float:
        """
        Calculate estimated days until reorder point is reached.

        Args:
            current_inventory: Current inventory level
            reorder_point: Reorder point
            demand_mean: Average daily demand

        Returns:
            Estimated days until reorder point
        """
        if demand_mean <= 0:
            return float('inf')

        if current_inventory <= reorder_point:
            return 0

        days_to_reorder = (current_inventory - reorder_point) / demand_mean
        return max(0, days_to_reorder)

    def simulate_inventory_policy(
        self,
        demand_series: pd.Series,
        reorder_point: float,
        order_quantity: float,
        lead_time: int = None,
        initial_inventory: float = None
    ) -> pd.DataFrame:
        """
        Simulate inventory levels under (R, Q) policy.

        Args:
            demand_series: Time series of demand
            reorder_point: Reorder point
            order_quantity: Order quantity
            lead_time: Lead time in days
            initial_inventory: Starting inventory (defaults to ROP + Q)

        Returns:
            DataFrame with simulation results
        """
        lt = lead_time if lead_time is not None else self.lead_time

        # Initialize
        if initial_inventory is None:
            initial_inventory = reorder_point + order_quantity

        inventory = [initial_inventory]
        orders = [0]
        stockouts = [0]
        pending_orders = []

        for t in range(len(demand_series)):
            # Check for incoming orders
            arriving_orders = [qty for day, qty in pending_orders if day == t]
            inventory_level = inventory[-1] + sum(arriving_orders)

            # Remove arrived orders
            pending_orders = [(day, qty) for day, qty in pending_orders if day != t]

            # Meet demand
            demand = demand_series.iloc[t]
            if inventory_level >= demand:
                inventory_level -= demand
                stockout = 0
            else:
                stockout = demand - inventory_level
                inventory_level = 0

            # Check if order should be placed
            if inventory_level <= reorder_point and len(pending_orders) == 0:
                pending_orders.append((t + lt, order_quantity))
                order_placed = order_quantity
            else:
                order_placed = 0

            inventory.append(inventory_level)
            orders.append(order_placed)
            stockouts.append(stockout)

        # Create results DataFrame
        results = pd.DataFrame({
            'period': range(len(inventory)),
            'inventory': inventory,
            'orders': [0] + orders,
            'stockouts': [0] + stockouts,
            'demand': [0] + list(demand_series)
        })

        # Calculate metrics
        results['service_level'] = ((results['demand'] - results['stockouts']) / results['demand']).fillna(1)

        logger.info("Simulation complete: %d periods", len(results))
        logger.info("Average inventory: %.2f", results['inventory'].mean())
        logger.info("Stockout rate: %.2f%%", (results['stockouts'] > 0).sum() / len(results) * 100)
        logger.info("Service level: %.2f%%", results['service_level'].mean() * 100)

        return results
