"""
Data Models for Supply Chain Planning System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class DemandResult:
    """Result from demand forecasting module."""
    
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    mape: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    elasticity: Optional[float] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    model_used: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class InventoryResult:
    """Result from inventory optimization module."""
    
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    classifications: pd.DataFrame = field(default_factory=pd.DataFrame)
    service_level: float = 0.0
    total_inventory_value: float = 0.0
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    safety_stock: pd.DataFrame = field(default_factory=pd.DataFrame)
    reorder_points: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PricingResult:
    """Result from dynamic pricing module."""
    
    optimal_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    revenue_lift: float = 0.0
    margin_improvement: float = 0.0
    elasticity_estimates: Dict[str, float] = field(default_factory=dict)
    markdown_recommendations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NetworkResult:
    """Result from network optimization module."""
    
    facility_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    routes: pd.DataFrame = field(default_factory=pd.DataFrame)
    capacity: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_reduction: float = 0.0
    distance_savings: float = 0.0
    utilization: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensingResult:
    """Result from real-time sensing module."""
    
    current_demand: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    short_term_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    trend_indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplenishmentResult:
    """Result from auto-replenishment module."""
    
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    purchase_orders: Optional[pd.DataFrame] = None
    transfer_orders: Optional[pd.DataFrame] = None
    automation_rate: float = 0.0
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    service_level_achieved: float = 0.0
    classifications: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PlanningResult:
    """Unified result from complete planning cycle."""
    
    planning_date: datetime = field(default_factory=datetime.now)
    horizon: str = ""
    
    # Module results
    demand: Optional[DemandResult] = None
    inventory: Optional[InventoryResult] = None
    pricing: Optional[PricingResult] = None
    network: Optional[NetworkResult] = None
    sensing: Optional[SensingResult] = None
    replenishment: Optional[ReplenishmentResult] = None
    
    # Aggregated KPIs
    kpis: Dict[str, float] = field(default_factory=dict)
    
    # Execution metadata
    execution_time_seconds: float = 0.0
    modules_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of planning results."""
        return {
            'planning_date': self.planning_date.isoformat(),
            'horizon': self.horizon,
            'modules_executed': self.modules_executed,
            'demand_mape': self.demand.mape if self.demand else None,
            'inventory_service_level': self.inventory.service_level if self.inventory else None,
            'pricing_revenue_lift': self.pricing.revenue_lift if self.pricing else None,
            'network_cost_reduction': self.network.cost_reduction if self.network else None,
            'replenishment_automation': self.replenishment.automation_rate if self.replenishment else None,
            'kpis': self.kpis,
            'errors': self.errors
        }
