"""Configuration schemas and validation.

This module defines the expected structure of configuration files
using dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ABCThresholds:
    """ABC classification thresholds.
    
    Items contributing to cumulative revenue up to threshold A are class A,
    A to B are class B, and the rest are class C.
    """
    A: float = 0.67  # Top items contributing to 67% of revenue
    B: float = 0.90  # Next items contributing to 67-90%
    # C is implicitly 90-100%


@dataclass
class XYZThresholds:
    """XYZ classification thresholds based on coefficient of variation."""
    X: float = 0.5   # Low variability (CV < 0.5)
    Y: float = 1.0   # Medium variability (0.5 <= CV < 1.0)
    # Z is implicitly CV >= 1.0


@dataclass
class ServiceLevelMatrix:
    """Service level targets for ABC-XYZ matrix combinations."""
    AX: float = 0.99
    AY: float = 0.97
    AZ: float = 0.95
    BX: float = 0.97
    BY: float = 0.95
    BZ: float = 0.92
    CX: float = 0.95
    CY: float = 0.92
    CZ: float = 0.90
    
    def get_service_level(self, abc_class: str, xyz_class: str) -> float:
        """Get service level for a classification combination.
        
        Args:
            abc_class: ABC class (A, B, or C)
            xyz_class: XYZ class (X, Y, or Z)
            
        Returns:
            Target service level
        """
        key = f"{abc_class}{xyz_class}"
        return getattr(self, key, 0.90)


@dataclass
class ClassificationConfig:
    """Configuration for item classification."""
    abc_thresholds: ABCThresholds = field(default_factory=ABCThresholds)
    xyz_thresholds: XYZThresholds = field(default_factory=XYZThresholds)
    service_levels: ServiceLevelMatrix = field(default_factory=ServiceLevelMatrix)
    
    # Column mappings
    item_id_column: str = "item_id"
    revenue_column: str = "revenue"
    quantity_column: str = "quantity"


@dataclass
class SafetyStockConfig:
    """Configuration for safety stock calculation."""
    method: str = "standard"  # standard, dynamic, or capacity_aware
    
    # Z-scores for common service levels
    z_scores: Dict[float, float] = field(default_factory=lambda: {
        0.90: 1.28,
        0.92: 1.41,
        0.95: 1.65,
        0.97: 1.88,
        0.99: 2.33,
    })
    
    # Capacity-aware parameters
    capacity_threshold: float = 0.85  # Utilization above which to increase SS
    capacity_multiplier: float = 1.2  # Multiplier when above threshold
    
    # Constraints
    min_safety_stock: Optional[int] = None
    max_safety_stock: Optional[int] = None


@dataclass
class PolicyConfig:
    """Configuration for replenishment policy."""
    type: str = "periodic_review"  # periodic_review (s,S), continuous_review (s,Q)
    review_period_days: int = 7
    order_strategy: str = "policy_target"  # policy_target, fill_to_capacity
    
    # Constraints
    min_order_quantity: Optional[int] = None
    max_order_quantity: Optional[int] = None
    order_multiple: Optional[int] = None


@dataclass
class AlertConfig:
    """Configuration for alert generation."""
    enabled: bool = True
    
    # Alert thresholds
    stockout_risk_threshold: float = 1.0  # Days of supply below which to alert
    excess_inventory_days: int = 90  # Days of supply above which is excess
    demand_spike_threshold: float = 0.5  # % increase to trigger spike alert
    
    # Alert severities
    severities: Dict[str, str] = field(default_factory=lambda: {
        "stockout_risk": "critical",
        "excess_inventory": "warning",
        "demand_spike": "warning",
        "trend_change": "info",
        "source_insufficient": "critical",
    })


@dataclass
class BinPackingConfig:
    """Configuration for 3D bin packing optimization."""
    enabled: bool = False
    
    # Orientation testing
    test_all_orientations: bool = True
    
    # Scoring weights
    utilization_weight: float = 0.4
    demand_match_weight: float = 0.3
    ergonomics_weight: float = 0.3
    
    # Ergonomic parameters
    max_item_weight_kg: float = 25.0
    preferred_height_range: tuple = (0.6, 1.4)  # meters from floor


@dataclass
class DemandAnalyticsConfig:
    """Configuration for demand analytics."""
    # Weighted moving average
    recency_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    lookback_periods: int = 4  # Number of periods for weighted average
    
    # Trend detection
    trend_threshold: float = 0.1  # % change to consider a trend
    trend_periods: int = 4  # Periods to evaluate for trend
    
    # Seasonality
    day_of_week_enabled: bool = True
    
    # Outlier handling
    outlier_method: str = "iqr"  # iqr, zscore
    outlier_threshold: float = 1.5  # IQR multiplier or z-score threshold


@dataclass
class ConfigSchema:
    """Main configuration schema."""
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    safety_stock: SafetyStockConfig = field(default_factory=SafetyStockConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    bin_packing: BinPackingConfig = field(default_factory=BinPackingConfig)
    demand_analytics: DemandAnalyticsConfig = field(default_factory=DemandAnalyticsConfig)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ConfigSchema":
        """Create ConfigSchema from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ConfigSchema instance
        """
        return cls(
            classification=ClassificationConfig(**config.get("classification", {})),
            safety_stock=SafetyStockConfig(**config.get("safety_stock", {})),
            policy=PolicyConfig(**config.get("policy", {})),
            alerts=AlertConfig(**config.get("alerts", {})),
            bin_packing=BinPackingConfig(**config.get("bin_packing", {})),
            demand_analytics=DemandAnalyticsConfig(**config.get("demand_analytics", {})),
        )


@dataclass
class ScenarioSchema:
    """Schema for scenario configuration."""
    name: str
    type: str  # external_supplier, internal_transfer, cross_dock, storage_to_picking
    
    # Source configuration
    source_type: str  # supplier, warehouse, storage_zone
    source_zone: Optional[str] = None
    lead_time_days: float = 7.0
    lead_time_variability: float = 0.0
    
    # Destination configuration
    destination_type: str = "warehouse"
    destination_zone: Optional[str] = None
    
    # Policy overrides
    review_period_days: int = 7
    order_strategy: str = "policy_target"
    
    # Constraints
    min_order_quantity: Optional[int] = None
    max_order_quantity: Optional[int] = None
    order_multiple: Optional[int] = None
    
    @classmethod
    def from_dict(cls, scenario: Dict[str, Any]) -> "ScenarioSchema":
        """Create ScenarioSchema from dictionary."""
        return cls(
            name=scenario.get("name", "Unnamed"),
            type=scenario.get("type", "external_supplier"),
            source_type=scenario.get("source", {}).get("type", "supplier"),
            source_zone=scenario.get("source", {}).get("zone"),
            lead_time_days=scenario.get("source", {}).get("lead_time_days", 7.0),
            lead_time_variability=scenario.get("source", {}).get("lead_time_variability", 0.0),
            destination_type=scenario.get("destination", {}).get("type", "warehouse"),
            destination_zone=scenario.get("destination", {}).get("zone"),
            review_period_days=scenario.get("policy", {}).get("review_period_days", 7),
            order_strategy=scenario.get("policy", {}).get("order_strategy", "policy_target"),
            min_order_quantity=scenario.get("constraints", {}).get("min_order_quantity"),
            max_order_quantity=scenario.get("constraints", {}).get("max_order_quantity"),
            order_multiple=scenario.get("constraints", {}).get("order_multiple"),
        )
