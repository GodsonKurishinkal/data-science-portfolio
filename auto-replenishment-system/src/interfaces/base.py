"""Abstract base classes defining core interfaces for the replenishment system.

These interfaces ensure consistent behavior and enable dependency injection
for testing and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class ILoader(ABC):
    """Interface for data loaders.

    Data loaders are responsible for loading data from various sources
    (CSV, database, API) and returning standardized DataFrames.
    """

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data from source.

        Args:
            **kwargs: Source-specific parameters (path, query, etc.)

        Returns:
            DataFrame containing loaded data

        Raises:
            DataLoadError: If data cannot be loaded
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the data source is accessible.

        Returns:
            True if source is accessible and valid
        """
        pass


class IValidator(ABC):
    """Interface for data validators.

    Validators check data quality and schema conformance.
    """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """List of required column names."""
        pass


class IClassifier(ABC):
    """Interface for item classifiers (ABC, XYZ, etc.).

    Classifiers categorize items based on various criteria
    for differentiated inventory management.
    """

    @abstractmethod
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items in the DataFrame.

        Args:
            df: DataFrame with item data

        Returns:
            DataFrame with classification column(s) added
        """
        pass

    @property
    @abstractmethod
    def classification_column(self) -> str:
        """Name of the classification column added."""
        pass


class IAnalyzer(ABC):
    """Interface for demand analyzers.

    Analyzers compute demand statistics, trends, and forecasts.
    """

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze demand data.

        Args:
            df: DataFrame with demand history

        Returns:
            DataFrame with analysis results
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from the analysis.

        Returns:
            Dictionary of computed statistics
        """
        pass


class IPolicy(ABC):
    """Interface for replenishment policies.

    Policies implement specific inventory management strategies
    like (s,S), (R,Q), (s,Q), etc.
    """

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate replenishment parameters.

        Args:
            df: DataFrame with inventory and demand data

        Returns:
            DataFrame with policy parameters (reorder point,
            order-up-to level, recommended quantity, etc.)
        """
        pass

    @property
    @abstractmethod
    def policy_type(self) -> str:
        """Type/name of the policy (e.g., 'periodic_review_sS')."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current policy parameters.

        Returns:
            Dictionary of policy configuration
        """
        pass


class IAlertGenerator(ABC):
    """Interface for alert generators.

    Alert generators identify inventory situations requiring attention.
    """

    @abstractmethod
    def generate_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate alerts based on inventory data.

        Args:
            df: DataFrame with inventory status and policy parameters

        Returns:
            List of alert dictionaries with keys:
                - alert_type: Type of alert (stockout_risk, excess, etc.)
                - severity: Alert severity (critical, warning, info)
                - item_id: Affected item
                - message: Human-readable description
                - data: Additional context data
        """
        pass

    @property
    @abstractmethod
    def alert_types(self) -> List[str]:
        """List of alert types this generator can produce."""
        pass


class IBinPacker(ABC):
    """Interface for 3D bin packing optimizers.

    Bin packers optimize how items are arranged in storage bins.
    """

    @abstractmethod
    def calculate_max_quantity(
        self,
        item_dimensions: Tuple[float, float, float],
        bin_dimensions: Tuple[float, float, float],
    ) -> int:
        """Calculate maximum quantity that fits in a bin.

        Args:
            item_dimensions: (length, width, height) of item
            bin_dimensions: (length, width, height) of bin

        Returns:
            Maximum number of items that fit
        """
        pass

    @abstractmethod
    def optimize_bin_selection(
        self,
        item_id: str,
        quantity: int,
        available_bins: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select optimal bin(s) for storing items.

        Args:
            item_id: Item identifier
            quantity: Quantity to store
            available_bins: List of available bin options

        Returns:
            Dictionary with selected bin(s) and arrangement
        """
        pass


class ISafetyStockCalculator(ABC):
    """Interface for safety stock calculators.

    Safety stock calculators determine buffer inventory levels.
    """

    @abstractmethod
    def calculate(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time: float,
        service_level: float,
        **kwargs,
    ) -> float:
        """Calculate safety stock level.

        Args:
            demand_mean: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            service_level: Target service level (0-1)
            **kwargs: Additional method-specific parameters

        Returns:
            Safety stock quantity
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of the calculation method."""
        pass


class IConfigLoader(ABC):
    """Interface for configuration loaders.

    Configuration loaders read and parse configuration files.
    """

    @abstractmethod
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary of configuration values
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        pass
