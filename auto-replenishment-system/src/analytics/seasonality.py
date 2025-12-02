"""Seasonality analysis for demand patterns.

Identifies day-of-week, monthly, and other seasonal patterns
to improve demand forecasting accuracy.
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SeasonalityAnalyzer:
    """Analyze seasonal patterns in demand data.

    Identifies:
    - Day-of-week patterns
    - Monthly patterns
    - Holiday effects
    - Custom seasonal cycles

    Examples:
        >>> analyzer = SeasonalityAnalyzer()
        >>> factors = analyzer.calculate_seasonal_factors(demand_df)
    """

    def __init__(
        self,
        item_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
        min_weeks: int = 4,
    ):
        """Initialize seasonality analyzer.

        Args:
            item_column: Item identifier column
            date_column: Date column
            quantity_column: Quantity column
            min_weeks: Minimum weeks of data required
        """
        self.item_column = item_column
        self.date_column = date_column
        self.quantity_column = quantity_column
        self.min_weeks = min_weeks

    def calculate_dow_factors(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Calculate day-of-week seasonal factors.

        Args:
            df: DataFrame with demand history
            normalize: If True, factors average to 1.0

        Returns:
            DataFrame with DOW factors per item
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        df["day_of_week"] = df[self.date_column].dt.dayofweek
        df["week"] = df[self.date_column].dt.isocalendar().week

        results = []

        for item_id in df[self.item_column].unique():
            item_data = df[df[self.item_column] == item_id]

            # Check minimum data requirement
            n_weeks = item_data["week"].nunique()
            if n_weeks < self.min_weeks:
                continue

            # Calculate average demand by day of week
            dow_avg = (
                item_data.groupby("day_of_week")[self.quantity_column]
                .mean()
                .to_dict()
            )

            # Normalize if requested
            if normalize and sum(dow_avg.values()) > 0:
                overall_mean = np.mean(list(dow_avg.values()))
                dow_avg = {k: v / overall_mean for k, v in dow_avg.items()}

            result = {self.item_column: item_id}
            for dow in range(7):
                day_name = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][dow]
                result[f"factor_{day_name}"] = dow_avg.get(dow, 1.0)

            results.append(result)

        return pd.DataFrame(results)

    def calculate_monthly_factors(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Calculate monthly seasonal factors.

        Args:
            df: DataFrame with demand history
            normalize: If True, factors average to 1.0

        Returns:
            DataFrame with monthly factors per item
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        df["month"] = df[self.date_column].dt.month
        df["year"] = df[self.date_column].dt.year

        results = []

        for item_id in df[self.item_column].unique():
            item_data = df[df[self.item_column] == item_id]

            # Need at least 6 months of data
            n_months = len(item_data.groupby(["year", "month"]))
            if n_months < 6:
                continue

            # Calculate average demand by month
            monthly_avg = (
                item_data.groupby("month")[self.quantity_column]
                .mean()
                .to_dict()
            )

            # Normalize if requested
            if normalize and sum(monthly_avg.values()) > 0:
                overall_mean = np.mean(list(monthly_avg.values()))
                monthly_avg = {k: v / overall_mean for k, v in monthly_avg.items()}

            result = {self.item_column: item_id}
            month_names = [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
            for month in range(1, 13):
                result[f"factor_{month_names[month-1]}"] = monthly_avg.get(month, 1.0)

            results.append(result)

        return pd.DataFrame(results)

    def apply_seasonal_factors(
        self,
        base_demand: float,
        factors: Dict[str, float],
        date: pd.Timestamp,
        factor_type: str = "dow",
    ) -> float:
        """Apply seasonal factors to base demand.

        Args:
            base_demand: Base demand value
            factors: Factor dictionary
            date: Date to apply factors for
            factor_type: Type of factor ('dow', 'monthly')

        Returns:
            Seasonally adjusted demand
        """
        if factor_type == "dow":
            dow = date.dayofweek
            day_name = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][dow]
            factor = factors.get(f"factor_{day_name}", 1.0)
        elif factor_type == "monthly":
            month = date.month
            month_names = [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
            factor = factors.get(f"factor_{month_names[month-1]}", 1.0)
        else:
            factor = 1.0

        return base_demand * factor

    def detect_seasonality_strength(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Measure the strength of seasonality for each item.

        Args:
            df: DataFrame with demand history

        Returns:
            DataFrame with seasonality strength metrics
        """
        dow_factors = self.calculate_dow_factors(df)
        monthly_factors = self.calculate_monthly_factors(df)

        results = []

        for item_id in df[self.item_column].unique():
            result = {self.item_column: item_id}

            # Calculate DOW seasonality strength (CV of factors)
            item_dow = dow_factors[
                dow_factors[self.item_column] == item_id
            ]
            if len(item_dow) > 0:
                dow_cols = [c for c in item_dow.columns if c.startswith("factor_")]
                dow_values = item_dow[dow_cols].values.flatten()
                result["dow_seasonality_strength"] = np.std(dow_values) / np.mean(dow_values) if np.mean(dow_values) > 0 else 0
            else:
                result["dow_seasonality_strength"] = 0

            # Calculate monthly seasonality strength
            item_monthly = monthly_factors[
                monthly_factors[self.item_column] == item_id
            ]
            if len(item_monthly) > 0:
                monthly_cols = [c for c in item_monthly.columns if c.startswith("factor_")]
                monthly_values = item_monthly[monthly_cols].values.flatten()
                result["monthly_seasonality_strength"] = np.std(monthly_values) / np.mean(monthly_values) if np.mean(monthly_values) > 0 else 0
            else:
                result["monthly_seasonality_strength"] = 0

            # Overall seasonality flag
            result["has_strong_seasonality"] = (
                result["dow_seasonality_strength"] > 0.2 or
                result["monthly_seasonality_strength"] > 0.3
            )

            results.append(result)

        return pd.DataFrame(results)


class HolidayAdjuster:
    """Adjust demand for holiday effects."""

    # US holidays with typical demand multipliers
    DEFAULT_HOLIDAYS = {
        "new_year": {"month": 1, "day": 1, "multiplier": 0.5},
        "memorial_day": {"month": 5, "day": -1, "multiplier": 1.3},  # Last Monday
        "independence_day": {"month": 7, "day": 4, "multiplier": 1.2},
        "labor_day": {"month": 9, "day": -1, "multiplier": 1.3},  # First Monday
        "thanksgiving": {"month": 11, "day": -1, "multiplier": 1.5},  # Fourth Thursday
        "christmas": {"month": 12, "day": 25, "multiplier": 0.3},
    }

    def __init__(
        self,
        holidays: Optional[Dict[str, Dict]] = None,
        pre_holiday_days: int = 3,
        post_holiday_days: int = 1,
    ):
        """Initialize holiday adjuster.

        Args:
            holidays: Custom holiday definitions
            pre_holiday_days: Days before holiday with adjusted demand
            post_holiday_days: Days after holiday with adjusted demand
        """
        self.holidays = holidays or self.DEFAULT_HOLIDAYS.copy()
        self.pre_holiday_days = pre_holiday_days
        self.post_holiday_days = post_holiday_days

    def get_holiday_factor(
        self,
        date: pd.Timestamp,
    ) -> float:
        """Get holiday adjustment factor for a date.

        Args:
            date: Date to check

        Returns:
            Multiplier factor (1.0 if no holiday effect)
        """
        # Check each holiday
        for holiday_name, config in self.holidays.items():
            holiday_date = self._get_holiday_date(date.year, config)

            if holiday_date is None:
                continue

            days_diff = (date.date() - holiday_date).days

            if days_diff == 0:
                return config["multiplier"]
            elif -self.pre_holiday_days <= days_diff < 0:
                # Pre-holiday period (typically higher)
                return 1.0 + (config["multiplier"] - 1.0) * 0.5
            elif 0 < days_diff <= self.post_holiday_days:
                # Post-holiday period
                return 1.0 + (config["multiplier"] - 1.0) * 0.3

        return 1.0

    def _get_holiday_date(
        self,
        year: int,
        config: Dict,
    ) -> Optional[pd.Timestamp]:
        """Get the actual date of a holiday for a given year."""
        import datetime

        month = config["month"]
        day = config.get("day", 1)

        if day > 0:
            # Fixed date holiday
            try:
                return datetime.date(year, month, day)
            except ValueError:
                return None
        else:
            # Floating holiday (negative day means nth weekday)
            # This is simplified - would need more logic for accurate floating holidays
            return datetime.date(year, month, 15)  # Placeholder
