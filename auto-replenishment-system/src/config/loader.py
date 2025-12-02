"""Configuration loader for YAML-based configuration files.

This module provides functionality to load and validate configuration
from YAML files, supporting both main config and scenario-specific configs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage YAML configuration files.

    Supports hierarchical configuration with scenario-specific overrides.

    Examples:
        >>> config = ConfigLoader.load("config/config.yaml")
        >>> scenario = ConfigLoader.load_scenario("config/scenarios/supplier.yaml")
    """

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing configuration values

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info("Loaded configuration from %s", config_path)
        return config

    @staticmethod
    def load_scenario(scenario_path: str) -> "ScenarioConfig":
        """Load a scenario configuration.

        Args:
            scenario_path: Path to the scenario YAML file

        Returns:
            ScenarioConfig object with validated configuration
        """
        config = ConfigLoader.load(scenario_path)
        return ScenarioConfig.from_dict(config)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate main configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Check required sections
        required_sections = ["classification", "safety_stock", "policy", "alerts"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: {section}")

        # Validate classification thresholds
        if "classification" in config:
            cls_config = config["classification"]
            if "abc_thresholds" in cls_config:
                thresholds = cls_config["abc_thresholds"]
                if not (0 < thresholds.get("A", 0) < thresholds.get("B", 0) <= 1.0):
                    errors.append("ABC thresholds must be: 0 < A < B <= 1.0")

        # Validate service levels
        if "service_levels" in config:
            for key, value in config["service_levels"].items():
                if not (0 < value <= 1.0):
                    errors.append(f"Service level {key} must be between 0 and 1")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge two configurations with override taking precedence.

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value

        return result


class ScenarioConfig:
    """Configuration for a specific replenishment scenario.

    Scenarios define source/destination zones, lead times, review periods,
    business rules, and constraints for a specific replenishment use case.

    Attributes:
        name: Scenario name
        scenario_type: Type of scenario (external_supplier, internal_transfer, cross_dock)
        source: Source configuration (supplier, zone, etc.)
        destination: Destination configuration
        policy: Replenishment policy settings
        constraints: Business constraints
    """

    def __init__(
        self,
        name: str,
        scenario_type: str,
        source: Dict[str, Any],
        destination: Dict[str, Any],
        policy: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        business_rules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize scenario configuration.

        Args:
            name: Scenario name
            scenario_type: Type of scenario
            source: Source configuration
            destination: Destination configuration
            policy: Policy settings
            constraints: Optional business constraints
            business_rules: Optional business rules
        """
        self.name = name
        self.scenario_type = scenario_type
        self.source = source
        self.destination = destination
        self.policy = policy
        self.constraints = constraints or {}
        self.business_rules = business_rules or {}

        self._validate()

    def _validate(self) -> None:
        """Validate scenario configuration."""
        valid_types = ["external_supplier", "internal_transfer", "cross_dock", "storage_to_picking"]
        if self.scenario_type not in valid_types:
            raise ValueError(f"Invalid scenario_type: {self.scenario_type}. Must be one of {valid_types}")

        # Validate source has required fields
        if "lead_time_days" not in self.source:
            raise ValueError("Source must specify lead_time_days")

        # Validate policy has required fields
        if "type" not in self.policy:
            raise ValueError("Policy must specify type")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ScenarioConfig":
        """Create ScenarioConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ScenarioConfig instance
        """
        scenario = config.get("scenario", config)

        return cls(
            name=scenario.get("name", "Unnamed Scenario"),
            scenario_type=scenario.get("type", "external_supplier"),
            source=scenario.get("source", {}),
            destination=scenario.get("destination", {}),
            policy=scenario.get("policy", {}),
            constraints=scenario.get("constraints"),
            business_rules=scenario.get("business_rules"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of scenario
        """
        return {
            "scenario": {
                "name": self.name,
                "type": self.scenario_type,
                "source": self.source,
                "destination": self.destination,
                "policy": self.policy,
                "constraints": self.constraints,
                "business_rules": self.business_rules,
            }
        }

    @property
    def lead_time(self) -> float:
        """Get lead time in days."""
        return self.source.get("lead_time_days", 7)

    @property
    def lead_time_variability(self) -> float:
        """Get lead time variability (standard deviation in days)."""
        return self.source.get("lead_time_variability", 0.0)

    @property
    def review_period(self) -> int:
        """Get review period in days."""
        return self.policy.get("review_period_days", 7)

    @property
    def order_strategy(self) -> str:
        """Get order quantity strategy."""
        return self.policy.get("order_strategy", "policy_target")

    @property
    def min_order_quantity(self) -> Optional[int]:
        """Get minimum order quantity constraint."""
        return self.constraints.get("min_order_quantity")

    @property
    def max_order_quantity(self) -> Optional[int]:
        """Get maximum order quantity constraint."""
        return self.constraints.get("max_order_quantity")

    @property
    def order_multiple(self) -> Optional[int]:
        """Get order multiple constraint."""
        return self.constraints.get("order_multiple")

    def __repr__(self) -> str:
        return f"ScenarioConfig(name='{self.name}', type='{self.scenario_type}')"
