"""Safety stock calculation module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SafetyStockCalculator:
    """
    Calculate safety stock levels using various methods.

    Safety stock is buffer inventory held to protect against demand and supply variability.
    """

    def __init__(
        self,
        service_level: float = 0.95,
        lead_time: int = 7
    ):
        """
        Initialize SafetyStockCalculator.

        Args:
            service_level: Target service level (e.g., 0.95 for 95%)
            lead_time: Lead time in days
        """
        self.service_level = service_level
        self.lead_time = lead_time
        self.z_score = stats.norm.ppf(service_level)

    def calculate_basic_safety_stock(
        self,
        demand_std: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate safety stock using basic formula.

        SS = Z * σ_demand * √lead_time

        Args:
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days (optional, uses default if not provided)

        Returns:
            Safety stock quantity
        """
        lt = lead_time if lead_time is not None else self.lead_time
        safety_stock = self.z_score * demand_std * np.sqrt(lt)
        return max(0, safety_stock)

    def calculate_safety_stock_with_lead_time_variability(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time_mean: float,
        lead_time_std: float
    ) -> float:
        """
        Calculate safety stock considering both demand and lead time variability.

        SS = Z * √(LT_mean * σ_demand² + μ_demand² * σ_LT²)

        Args:
            demand_mean: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time_mean: Average lead time in days
            lead_time_std: Standard deviation of lead time

        Returns:
            Safety stock quantity
        """
        variance = (
            lead_time_mean * (demand_std ** 2) +
            (demand_mean ** 2) * (lead_time_std ** 2)
        )
        safety_stock = self.z_score * np.sqrt(variance)
        return max(0, safety_stock)

    def calculate_periodic_review_safety_stock(
        self,
        demand_std: float,
        lead_time: int = None,
        review_period: int = 7
    ) -> float:
        """
        Calculate safety stock for periodic review system.

        SS = Z * σ_demand * √(lead_time + review_period)

        Args:
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            review_period: Review period in days

        Returns:
            Safety stock quantity
        """
        lt = lead_time if lead_time is not None else self.lead_time
        protection_period = lt + review_period
        safety_stock = self.z_score * demand_std * np.sqrt(protection_period)
        return max(0, safety_stock)

    def calculate_with_forecast_error(
        self,
        forecast_std: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate safety stock based on forecast error.

        SS = Z * σ_forecast * √lead_time

        Args:
            forecast_std: Standard deviation of forecast error
            lead_time: Lead time in days

        Returns:
            Safety stock quantity
        """
        lt = lead_time if lead_time is not None else self.lead_time
        safety_stock = self.z_score * forecast_std * np.sqrt(lt)
        return max(0, safety_stock)

    def calculate_for_dataframe(
        self,
        data: pd.DataFrame,
        method: str = 'basic',
        _demand_mean_col: str = 'sales_mean',
        demand_std_col: str = 'sales_std',
        lead_time_col: str = None,
        review_period: int = 7
    ) -> pd.DataFrame:
        """
        Calculate safety stock for multiple items in a DataFrame.

        Args:
            data: DataFrame with demand statistics
            method: Calculation method ('basic', 'periodic', 'forecast')
            demand_mean_col: Column name for mean demand
            demand_std_col: Column name for demand std dev
            lead_time_col: Column name for lead time (optional)
            review_period: Review period for periodic method

        Returns:
            DataFrame with safety stock column added
        """
        logger.info("Calculating safety stock using method: %s", method)

        result = data.copy()

        if method == 'basic':
            if lead_time_col and lead_time_col in result.columns:
                result['safety_stock'] = result.apply(
                    lambda row: self.calculate_basic_safety_stock(
                        row[demand_std_col],
                        row[lead_time_col]
                    ),
                    axis=1
                )
            else:
                result['safety_stock'] = result[demand_std_col].apply(
                    self.calculate_basic_safety_stock
                )

        elif method == 'periodic':
            if lead_time_col and lead_time_col in result.columns:
                result['safety_stock'] = result.apply(
                    lambda row: self.calculate_periodic_review_safety_stock(
                        row[demand_std_col],
                        row[lead_time_col],
                        review_period
                    ),
                    axis=1
                )
            else:
                result['safety_stock'] = result[demand_std_col].apply(
                    lambda std: self.calculate_periodic_review_safety_stock(
                        std, review_period=review_period
                    )
                )

        elif method == 'forecast':
            result['safety_stock'] = result[demand_std_col].apply(
                self.calculate_with_forecast_error
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Round to whole units
        result['safety_stock'] = result['safety_stock'].round(0).astype(int)

        logger.info("Average safety stock: %.2f", result['safety_stock'].mean())
        logger.info("Total safety stock: %.0f", result['safety_stock'].sum())

        return result

    def calculate_by_service_level(
        self,
        demand_std: float,
        lead_time: int = None,
        service_levels: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate safety stock for multiple service levels.

        Args:
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            service_levels: Dictionary of service level names and values

        Returns:
            Dictionary of safety stock quantities by service level
        """
        if service_levels is None:
            service_levels = {
                '85%': 0.85,
                '90%': 0.90,
                '95%': 0.95,
                '99%': 0.99
            }

        lt = lead_time if lead_time is not None else self.lead_time
        results = {}

        for name, sl in service_levels.items():
            z = stats.norm.ppf(sl)
            ss = z * demand_std * np.sqrt(lt)
            results[name] = max(0, ss)

        return results

    def estimate_stockout_probability(
        self,
        current_stock: float,
        demand_mean: float,
        demand_std: float,
        lead_time: int = None
    ) -> float:
        """
        Estimate probability of stockout given current inventory.

        Args:
            current_stock: Current inventory level
            demand_mean: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days

        Returns:
            Probability of stockout (0 to 1)
        """
        lt = lead_time if lead_time is not None else self.lead_time

        # Expected demand during lead time
        expected_demand = demand_mean * lt

        # Standard deviation of demand during lead time
        demand_std_lt = demand_std * np.sqrt(lt)

        if demand_std_lt == 0:
            return 0.0 if current_stock >= expected_demand else 1.0

        # Calculate z-score
        z = (current_stock - expected_demand) / demand_std_lt

        # Probability of stockout (demand exceeds current stock)
        stockout_prob = 1 - stats.norm.cdf(z)

        return stockout_prob

    def calculate_optimal_service_level(
        self,
        unit_cost: float,
        holding_cost_rate: float,
        stockout_cost: float,
        _demand_std: float,
        lead_time: int = None
    ) -> float:
        """
        Calculate economically optimal service level.

        Args:
            unit_cost: Cost per unit
            holding_cost_rate: Annual holding cost rate (e.g., 0.25 for 25%)
            stockout_cost: Cost per stockout occurrence
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days

        Returns:
            Optimal service level (0 to 1)
        """
        lt = lead_time if lead_time is not None else self.lead_time

        # Annual holding cost per unit
        annual_holding_cost = unit_cost * holding_cost_rate

        # Daily holding cost per unit
        daily_holding_cost = annual_holding_cost / 365

        # Cost of holding safety stock for lead time
        holding_cost = daily_holding_cost * lt

        # Critical ratio
        critical_ratio = stockout_cost / (stockout_cost + holding_cost)

        # Optimal service level
        optimal_sl = min(0.99, max(0.50, critical_ratio))

        return optimal_sl
