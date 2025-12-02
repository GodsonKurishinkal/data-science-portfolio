"""Universal Replenishment Engine.

Main orchestrator that coordinates:
- Data loading and validation
- Classification (ABC/XYZ/Velocity)
- Demand analytics
- Safety stock calculation
- Policy-based order recommendations
- Alert generation

Supports all retail replenishment scenarios through configuration.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import numpy as np

from ..config.loader import ConfigLoader, ScenarioConfig
from ..data.loaders import InventoryDataLoader, DemandDataLoader, SourceInventoryLoader
from ..data.validators import InventoryValidator, DemandValidator
from ..classification.abc_classifier import ABCClassifier
from ..classification.xyz_classifier import XYZClassifier
from ..classification.velocity_classifier import VelocityClassifier
from ..classification.matrix import ABCXYZMatrix
from ..analytics.demand import DemandAnalyzer
from ..safety_stock.calculator import SafetyStockCalculator
from ..policies.periodic_review import PeriodicReviewPolicy
from ..policies.continuous_review import ContinuousReviewPolicy
from ..policies.min_max import MinMaxPolicy
from ..alerts.generator import AlertGenerator, AlertThresholds
from ..alerts.types import Alert

logger = logging.getLogger(__name__)


@dataclass
class ReplenishmentResult:
    """Container for replenishment engine results.

    Attributes:
        scenario: Scenario configuration used
        recommendations: DataFrame with order recommendations
        alerts: List of generated alerts
        summary: Summary statistics
        classification: Classification results
        analytics: Demand analytics results
        timestamp: Execution timestamp
    """
    scenario: str
    recommendations: pd.DataFrame
    alerts: List[Alert]
    summary: Dict[str, Any]
    classification: Dict[str, pd.DataFrame] = field(default_factory=dict)
    analytics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario": self.scenario,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "recommendations_count": len(self.recommendations),
            "alerts_count": len(self.alerts),
        }

    def get_priority_orders(
        self,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top priority order recommendations.

        Args:
            top_n: Number of top priority items

        Returns:
            DataFrame with priority orders
        """
        if self.recommendations.empty:
            return self.recommendations

        # Filter items that need orders
        needs_order = self.recommendations[
            self.recommendations.get("needs_order", True) == True  # noqa
        ]

        # Sort by criticality (days of supply ascending)
        if "days_of_supply" in needs_order.columns:
            return needs_order.nsmallest(top_n, "days_of_supply")

        return needs_order.head(top_n)


class ReplenishmentEngine:
    """Universal Replenishment Engine for all retail scenarios.

    Features:
    - 100% YAML configuration driven
    - Supports multiple scenarios without code changes
    - Multi-policy support (s,S, s,Q, min-max)
    - Integrated classification and analytics
    - Actionable alerts generation

    Scenarios supported:
    - Supplier to DC
    - DC to Store
    - Store to DC (returns)
    - Storage to Picking
    - Backroom to Shelf
    - Cross-dock
    - Inter-store transfers
    - E-commerce fulfillment

    Examples:
        >>> engine = ReplenishmentEngine(config_path="config/config.yaml")
        >>> result = engine.run(
        ...     scenario="dc_to_store",
        ...     inventory_data=inventory_df,
        ...     demand_data=demand_df
        ... )
        >>> print(result.summary)
    """

    POLICY_REGISTRY: Dict[str, Type] = {
        "periodic_review": PeriodicReviewPolicy,
        "periodic_review_sS": PeriodicReviewPolicy,
        "continuous_review": ContinuousReviewPolicy,
        "continuous_review_sQ": ContinuousReviewPolicy,
        "min_max": MinMaxPolicy,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialize replenishment engine.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        self.config_path = config_path
        self.config_dict = config_dict

        # Load configuration
        if config_path:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.load()
        else:
            self.config = config_dict or {}

        # Initialize components (lazy loaded per scenario)
        self._classifiers: Dict[str, Any] = {}
        self._policies: Dict[str, Any] = {}
        self._analyzers: Dict[str, Any] = {}

        logger.info("ReplenishmentEngine initialized")

    def run(
        self,
        scenario: str,
        inventory_data: pd.DataFrame,
        demand_data: Optional[pd.DataFrame] = None,
        source_inventory: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ReplenishmentResult:
        """Execute replenishment calculation for a scenario.

        Args:
            scenario: Scenario name (must be defined in config)
            inventory_data: Current inventory positions
            demand_data: Historical demand data (for analytics)
            source_inventory: Source location inventory (for availability)
            **kwargs: Additional parameters to override config

        Returns:
            ReplenishmentResult with recommendations and alerts
        """
        logger.info(f"Running replenishment for scenario: {scenario}")

        # Get scenario configuration
        scenario_config = self._get_scenario_config(scenario, **kwargs)

        # Validate input data
        inventory_data = self._validate_inventory(
            inventory_data, scenario_config
        )

        # Perform classification
        classification_results = self._run_classification(
            inventory_data, demand_data, scenario_config
        )

        # Run demand analytics
        analytics_results = {}
        if demand_data is not None and not demand_data.empty:
            analytics_results = self._run_analytics(
                demand_data, scenario_config
            )
            # Merge analytics into inventory data
            inventory_data = self._merge_analytics(
                inventory_data, analytics_results
            )

        # Merge classification into inventory data
        inventory_data = self._merge_classification(
            inventory_data, classification_results, scenario_config
        )

        # Add source availability
        if source_inventory is not None:
            inventory_data = self._merge_source_availability(
                inventory_data, source_inventory, scenario_config
            )

        # Calculate replenishment recommendations
        policy = self._get_policy(scenario_config)
        recommendations = policy.calculate(inventory_data)

        # Generate alerts
        alert_generator = self._get_alert_generator(scenario_config)
        alerts = alert_generator.generate(recommendations)

        # Create summary
        summary = self._create_summary(
            recommendations, alerts, scenario_config
        )

        return ReplenishmentResult(
            scenario=scenario,
            recommendations=recommendations,
            alerts=alerts,
            summary=summary,
            classification=classification_results,
            analytics=analytics_results,
        )

    def _get_scenario_config(
        self,
        scenario: str,
        **overrides,
    ) -> Dict[str, Any]:
        """Get configuration for a scenario."""
        scenarios = self.config.get("scenarios", {})

        if scenario not in scenarios:
            logger.warning(
                f"Scenario '{scenario}' not in config, using defaults"
            )
            scenario_config = self._get_default_config()
        else:
            scenario_config = scenarios[scenario].copy()

        # Apply overrides
        scenario_config.update(overrides)

        return scenario_config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "policy_type": "periodic_review",
            "review_period": 7,
            "lead_time": 7,
            "service_level": 0.95,
            "order_strategy": "policy_target",
            "classification": {
                "abc_enabled": True,
                "xyz_enabled": True,
            },
            "alerts": {
                "enabled": True,
            },
        }

    def _validate_inventory(
        self,
        inventory_data: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Validate and preprocess inventory data."""
        validator = InventoryValidator(
            required_columns=config.get("required_columns", [
                "item_id", "current_stock"
            ])
        )

        return validator.validate(inventory_data)

    def _run_classification(
        self,
        inventory_data: pd.DataFrame,
        demand_data: Optional[pd.DataFrame],
        config: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """Run ABC/XYZ/Velocity classification."""
        results = {}
        classification_config = config.get("classification", {})

        # ABC Classification
        if classification_config.get("abc_enabled", True):
            abc_config = classification_config.get("abc", {})
            abc_classifier = ABCClassifier(
                value_column=abc_config.get("value_column", "revenue"),
                thresholds=abc_config.get("thresholds", (0.80, 0.95)),
            )

            if "revenue" in inventory_data.columns:
                results["abc"] = abc_classifier.classify(inventory_data)

        # XYZ Classification
        if classification_config.get("xyz_enabled", True) and demand_data is not None:
            xyz_config = classification_config.get("xyz", {})
            xyz_classifier = XYZClassifier(
                cv_thresholds=xyz_config.get("thresholds", (0.5, 1.0)),
            )

            results["xyz"] = xyz_classifier.classify(demand_data)

        # Velocity Classification
        if classification_config.get("velocity_enabled", False):
            velocity_config = classification_config.get("velocity", {})
            velocity_classifier = VelocityClassifier(
                thresholds=velocity_config.get("thresholds", (0.6, 0.9)),
            )

            if "daily_demand_rate" in inventory_data.columns:
                results["velocity"] = velocity_classifier.classify(inventory_data)

        return results

    def _run_analytics(
        self,
        demand_data: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run demand analytics."""
        analytics_config = config.get("analytics", {})

        analyzer = DemandAnalyzer(
            date_column=analytics_config.get("date_column", "date"),
            demand_column=analytics_config.get("demand_column", "quantity"),
            item_column=analytics_config.get("item_column", "item_id"),
        )

        return {
            "demand_stats": analyzer.calculate_statistics(demand_data),
        }

    def _merge_analytics(
        self,
        inventory_data: pd.DataFrame,
        analytics_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Merge analytics results into inventory data."""
        if "demand_stats" in analytics_results:
            stats_df = analytics_results["demand_stats"]
            if not stats_df.empty:
                # Merge on item_id
                item_col = "item_id"
                if item_col in inventory_data.columns and item_col in stats_df.columns:
                    inventory_data = inventory_data.merge(
                        stats_df,
                        on=item_col,
                        how="left",
                        suffixes=("", "_analytics"),
                    )

        return inventory_data

    def _merge_classification(
        self,
        inventory_data: pd.DataFrame,
        classification_results: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Merge classification results into inventory data."""
        # Create ABC-XYZ matrix for service levels
        if "abc" in classification_results and "xyz" in classification_results:
            matrix = ABCXYZMatrix()
            service_levels = config.get("service_level_matrix", {})

            if service_levels:
                matrix.set_service_levels(service_levels)

            # Add target service level based on classification
            abc_df = classification_results["abc"]
            xyz_df = classification_results["xyz"]

            if "abc_class" in abc_df.columns:
                inventory_data = inventory_data.merge(
                    abc_df[["item_id", "abc_class"]],
                    on="item_id",
                    how="left",
                )

            if "xyz_class" in xyz_df.columns:
                inventory_data = inventory_data.merge(
                    xyz_df[["item_id", "xyz_class"]],
                    on="item_id",
                    how="left",
                )

            # Calculate target service level from matrix
            if "abc_class" in inventory_data.columns and "xyz_class" in inventory_data.columns:
                inventory_data["target_service_level"] = inventory_data.apply(
                    lambda row: matrix.get_service_level(
                        row.get("abc_class", "B"),
                        row.get("xyz_class", "Y"),
                    ),
                    axis=1,
                )

        return inventory_data

    def _merge_source_availability(
        self,
        inventory_data: pd.DataFrame,
        source_inventory: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Merge source inventory availability."""
        item_col = config.get("item_column", "item_id")
        source_qty_col = config.get("source_quantity_column", "available_quantity")

        if item_col in source_inventory.columns:
            source_df = source_inventory[[item_col, source_qty_col]].copy()
            source_df.columns = [item_col, "source_available"]

            inventory_data = inventory_data.merge(
                source_df,
                on=item_col,
                how="left",
            )

            # Fill missing with infinity (no constraint)
            inventory_data["source_available"] = inventory_data["source_available"].fillna(
                float("inf")
            )

        return inventory_data

    def _get_policy(self, config: Dict[str, Any]):
        """Get or create policy based on configuration."""
        policy_type = config.get("policy_type", "periodic_review")

        policy_class = self.POLICY_REGISTRY.get(policy_type)
        if policy_class is None:
            logger.warning(
                f"Unknown policy type '{policy_type}', using periodic_review"
            )
            policy_class = PeriodicReviewPolicy

        # Extract policy parameters
        policy_params = {
            "review_period": config.get("review_period", 7),
            "lead_time": config.get("lead_time", 7),
            "service_level": config.get("service_level", 0.95),
            "order_strategy": config.get("order_strategy", "policy_target"),
        }

        # Handle policy-specific parameters
        if policy_type in ["periodic_review", "periodic_review_sS"]:
            return PeriodicReviewPolicy(**policy_params)
        elif policy_type in ["continuous_review", "continuous_review_sQ"]:
            return ContinuousReviewPolicy(
                lead_time=policy_params["lead_time"],
                service_level=policy_params["service_level"],
                ordering_cost=config.get("ordering_cost", 50),
                holding_cost_rate=config.get("holding_cost_rate", 0.25),
            )
        elif policy_type == "min_max":
            return MinMaxPolicy(
                min_days_supply=config.get("min_days_supply", 7),
                max_days_supply=config.get("max_days_supply", 21),
            )

        return policy_class(**policy_params)

    def _get_alert_generator(
        self,
        config: Dict[str, Any],
    ) -> AlertGenerator:
        """Get alert generator with configuration."""
        alert_config = config.get("alerts", {})

        thresholds = AlertThresholds(
            critical_days_supply=alert_config.get("critical_days_supply", 1.0),
            low_days_supply=alert_config.get("low_days_supply", 3.0),
            overstock_days_supply=alert_config.get("overstock_days_supply", 60.0),
        )

        return AlertGenerator(thresholds=thresholds)

    def _create_summary(
        self,
        recommendations: pd.DataFrame,
        alerts: List[Alert],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create execution summary."""
        if recommendations.empty:
            return {
                "total_items": 0,
                "items_needing_order": 0,
                "total_recommended_quantity": 0,
                "total_alerts": 0,
            }

        needs_order = recommendations.get("needs_order", pd.Series([False]))
        rec_qty = recommendations.get("recommended_quantity", pd.Series([0]))

        summary = {
            "total_items": len(recommendations),
            "items_needing_order": needs_order.sum() if not needs_order.empty else 0,
            "items_not_needing_order": (~needs_order).sum() if not needs_order.empty else 0,
            "total_recommended_quantity": rec_qty.sum(),
            "total_alerts": len(alerts),
            "critical_alerts": sum(
                1 for a in alerts if a.severity.value == "critical"
            ),
            "high_alerts": sum(
                1 for a in alerts if a.severity.value == "high"
            ),
        }

        # Add classification breakdown if available
        if "abc_class" in recommendations.columns:
            summary["by_abc_class"] = (
                recommendations["abc_class"].value_counts().to_dict()
            )

        if "xyz_class" in recommendations.columns:
            summary["by_xyz_class"] = (
                recommendations["xyz_class"].value_counts().to_dict()
            )

        return summary

    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenarios."""
        return list(self.config.get("scenarios", {}).keys())

    def validate_scenario(self, scenario: str) -> Dict[str, Any]:
        """Validate scenario configuration.

        Args:
            scenario: Scenario name

        Returns:
            Validation result dictionary
        """
        issues = []
        warnings = []

        scenarios = self.config.get("scenarios", {})
        if scenario not in scenarios:
            issues.append(f"Scenario '{scenario}' not found in configuration")
            return {
                "valid": False,
                "issues": issues,
                "warnings": warnings,
            }

        config = scenarios[scenario]

        # Check required fields
        required_fields = ["policy_type", "lead_time"]
        for field in required_fields:
            if field not in config:
                warnings.append(f"Missing field '{field}', will use default")

        # Check policy type
        policy_type = config.get("policy_type", "periodic_review")
        if policy_type not in self.POLICY_REGISTRY:
            issues.append(f"Unknown policy type: {policy_type}")

        # Check service level range
        sl = config.get("service_level", 0.95)
        if not 0 < sl < 1:
            issues.append(f"Service level must be between 0 and 1, got {sl}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
