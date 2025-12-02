"""
Supply Chain Planner - Master Orchestrator.

This module provides the main SupplyChainPlanner class that orchestrates
all planning functions across the supply chain.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import logging

import pandas as pd

from src.config.loader import PlanningConfig
from src.data.models import (
    PlanningResult,
    DemandResult,
    InventoryResult,
    PricingResult,
    NetworkResult,
    SensingResult,
    ReplenishmentResult,
)
from src.integrations.demand_integration import DemandIntegration
from src.integrations.inventory_integration import InventoryIntegration
from src.integrations.pricing_integration import PricingIntegration
from src.integrations.network_integration import NetworkIntegration
from src.integrations.sensing_integration import SensingIntegration
from src.integrations.replenishment_integration import ReplenishmentIntegration
from src.kpi.calculator import KPICalculator

logger = logging.getLogger(__name__)


@dataclass
class SOPPlan:
    """Sales & Operations Planning result."""
    
    planning_date: date
    horizon_months: int
    demand_plan: pd.DataFrame
    supply_plan: pd.DataFrame
    inventory_plan: pd.DataFrame
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    scenarios: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def analyze_gaps(self) -> List[Dict[str, Any]]:
        """Analyze demand-supply gaps."""
        return self.gaps
    
    def get_recommendations(self) -> List[str]:
        """Get planning recommendations."""
        return self.recommendations


@dataclass
class DailyReplenishmentResult:
    """Daily replenishment run result."""
    
    run_date: date
    scenarios: List[str]
    purchase_orders: pd.DataFrame
    transfer_orders: pd.DataFrame
    alerts: List[Dict[str, Any]]
    
    def get_purchase_orders(self) -> pd.DataFrame:
        """Get generated purchase orders."""
        return self.purchase_orders
    
    def get_transfer_orders(self) -> pd.DataFrame:
        """Get generated transfer orders."""
        return self.transfer_orders


class SupplyChainPlanner:
    """
    Master orchestrator for end-to-end supply chain planning.
    
    Integrates all planning modules:
    - Demand Forecasting
    - Inventory Optimization
    - Dynamic Pricing
    - Network Optimization
    - Real-Time Sensing
    - Auto-Replenishment
    
    Example:
        >>> config = PlanningConfig.from_yaml('config/config.yaml')
        >>> planner = SupplyChainPlanner(config)
        >>> results = planner.run_planning_cycle(planning_horizon='monthly')
    """
    
    def __init__(self, config: PlanningConfig):
        """
        Initialize the Supply Chain Planner.
        
        Args:
            config: Planning configuration object
        """
        self.config = config
        self._initialize_integrations()
        self.kpi_calculator = KPICalculator(config)
        logger.info("SupplyChainPlanner initialized")
    
    def _initialize_integrations(self) -> None:
        """Initialize all module integrations."""
        self.demand = DemandIntegration(self.config.demand)
        self.inventory = InventoryIntegration(self.config.inventory)
        self.pricing = PricingIntegration(self.config.pricing)
        self.network = NetworkIntegration(self.config.network)
        self.sensing = SensingIntegration(self.config.sensing)
        self.replenishment = ReplenishmentIntegration(self.config.replenishment)
    
    def run_planning_cycle(
        self,
        planning_horizon: str = 'monthly',
        include_modules: Optional[List[str]] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> PlanningResult:
        """
        Run a complete planning cycle.
        
        Args:
            planning_horizon: Planning horizon ('daily', 'weekly', 'monthly')
            include_modules: List of modules to include (default: all)
            data: Optional input data for each module
            
        Returns:
            PlanningResult with results from all modules
        """
        if include_modules is None:
            include_modules = ['demand', 'inventory', 'pricing', 'network', 'sensing', 'replenishment']
        
        if data is None:
            data = {}
        
        logger.info(
            "Starting %s planning cycle with modules: %s",
            planning_horizon,
            include_modules
        )
        
        results = PlanningResult(
            planning_date=datetime.now(),
            horizon=planning_horizon
        )
        
        # Run modules in dependency order
        if 'demand' in include_modules:
            results.demand = self._run_demand_planning(data.get('demand'))
        
        if 'inventory' in include_modules:
            results.inventory = self._run_inventory_optimization(
                data.get('inventory'),
                forecast=results.demand.forecast if results.demand else None
            )
        
        if 'pricing' in include_modules:
            results.pricing = self._run_pricing_optimization(
                data.get('pricing'),
                demand_elasticity=results.demand.elasticity if results.demand else None
            )
        
        if 'network' in include_modules:
            results.network = self._run_network_optimization(
                data.get('network'),
                inventory_positions=results.inventory.positions if results.inventory else None
            )
        
        if 'sensing' in include_modules:
            results.sensing = self._run_demand_sensing(data.get('sensing'))
        
        if 'replenishment' in include_modules:
            results.replenishment = self._run_replenishment(
                data.get('replenishment'),
                inventory=results.inventory,
                forecast=results.demand.forecast if results.demand else None
            )
        
        # Calculate integrated KPIs
        results.kpis = self.kpi_calculator.calculate_all(results)
        
        logger.info("Planning cycle completed")
        return results
    
    def _run_demand_planning(self, data: Optional[pd.DataFrame]) -> DemandResult:
        """Run demand forecasting module."""
        logger.info("Running demand planning...")
        return self.demand.run(data)
    
    def _run_inventory_optimization(
        self,
        data: Optional[pd.DataFrame],
        forecast: Optional[pd.DataFrame] = None
    ) -> InventoryResult:
        """Run inventory optimization module."""
        logger.info("Running inventory optimization...")
        return self.inventory.run(data, forecast=forecast)
    
    def _run_pricing_optimization(
        self,
        data: Optional[pd.DataFrame],
        demand_elasticity: Optional[float] = None
    ) -> PricingResult:
        """Run pricing optimization module."""
        logger.info("Running pricing optimization...")
        return self.pricing.run(data, elasticity=demand_elasticity)
    
    def _run_network_optimization(
        self,
        data: Optional[pd.DataFrame],
        inventory_positions: Optional[pd.DataFrame] = None
    ) -> NetworkResult:
        """Run network optimization module."""
        logger.info("Running network optimization...")
        return self.network.run(data, inventory=inventory_positions)
    
    def _run_demand_sensing(self, data: Optional[pd.DataFrame]) -> SensingResult:
        """Run real-time demand sensing module."""
        logger.info("Running demand sensing...")
        return self.sensing.run(data)
    
    def _run_replenishment(
        self,
        data: Optional[pd.DataFrame],
        inventory: Optional[InventoryResult] = None,
        forecast: Optional[pd.DataFrame] = None
    ) -> ReplenishmentResult:
        """Run auto-replenishment module."""
        logger.info("Running auto-replenishment...")
        return self.replenishment.run(data, inventory=inventory, forecast=forecast)
    
    def generate_sop_plan(
        self,
        horizon_months: int = 3,
        scenarios: Optional[List[str]] = None
    ) -> SOPPlan:
        """
        Generate a Sales & Operations Planning (S&OP) plan.
        
        Args:
            horizon_months: Planning horizon in months
            scenarios: List of scenarios to evaluate (default: base, optimistic, pessimistic)
            
        Returns:
            SOPPlan with demand, supply, and inventory plans
        """
        if scenarios is None:
            scenarios = ['base', 'optimistic', 'pessimistic']
        
        logger.info(
            "Generating S&OP plan for %d months with scenarios: %s",
            horizon_months,
            scenarios
        )
        
        # Generate demand plan
        demand_result = self.demand.run(None, horizon_months=horizon_months)
        demand_plan = demand_result.forecast if demand_result else pd.DataFrame()
        
        # Generate supply plan
        network_result = self.network.run(None)
        supply_plan = network_result.capacity if network_result else pd.DataFrame()
        
        # Generate inventory plan
        inventory_result = self.inventory.run(None, forecast=demand_plan)
        inventory_plan = inventory_result.positions if inventory_result else pd.DataFrame()
        
        # Analyze gaps
        gaps = self._analyze_demand_supply_gaps(demand_plan, supply_plan)
        
        # Generate scenario results
        scenario_results = {}
        for scenario in scenarios:
            scenario_results[scenario] = self._evaluate_scenario(scenario, demand_plan)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gaps, scenario_results)
        
        return SOPPlan(
            planning_date=date.today(),
            horizon_months=horizon_months,
            demand_plan=demand_plan,
            supply_plan=supply_plan,
            inventory_plan=inventory_plan,
            gaps=gaps,
            scenarios=scenario_results,
            recommendations=recommendations
        )
    
    def run_daily_replenishment(
        self,
        run_date: str,
        scenarios: Optional[List[str]] = None
    ) -> DailyReplenishmentResult:
        """
        Run daily replenishment cycle.
        
        Args:
            run_date: Date for replenishment run (YYYY-MM-DD)
            scenarios: Replenishment scenarios to run
            
        Returns:
            DailyReplenishmentResult with orders and alerts
        """
        if scenarios is None:
            scenarios = ['dc_to_store', 'supplier_to_dc']
        
        logger.info(
            "Running daily replenishment for %s with scenarios: %s",
            run_date,
            scenarios
        )
        
        # Run replenishment for each scenario
        all_purchase_orders = []
        all_transfer_orders = []
        all_alerts = []
        
        for scenario in scenarios:
            result = self.replenishment.run_scenario(scenario, run_date)
            
            if result.purchase_orders is not None:
                all_purchase_orders.append(result.purchase_orders)
            if result.transfer_orders is not None:
                all_transfer_orders.append(result.transfer_orders)
            all_alerts.extend(result.alerts)
        
        # Combine results
        purchase_orders = pd.concat(all_purchase_orders) if all_purchase_orders else pd.DataFrame()
        transfer_orders = pd.concat(all_transfer_orders) if all_transfer_orders else pd.DataFrame()
        
        return DailyReplenishmentResult(
            run_date=datetime.strptime(run_date, '%Y-%m-%d').date(),
            scenarios=scenarios,
            purchase_orders=purchase_orders,
            transfer_orders=transfer_orders,
            alerts=all_alerts
        )
    
    def monitor_exceptions(self) -> List[Dict[str, Any]]:
        """
        Monitor for planning exceptions across all modules.
        
        Returns:
            List of exception dictionaries with severity and details
        """
        logger.info("Monitoring for exceptions...")
        
        exceptions = []
        
        # Check demand sensing for anomalies
        sensing_exceptions = self.sensing.get_active_alerts()
        exceptions.extend(sensing_exceptions)
        
        # Check inventory for stockouts/overstock
        inventory_exceptions = self.inventory.get_exceptions()
        exceptions.extend(inventory_exceptions)
        
        # Check replenishment for urgent needs
        replenishment_exceptions = self.replenishment.get_urgent_items()
        exceptions.extend(replenishment_exceptions)
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        exceptions.sort(key=lambda x: severity_order.get(x.get('severity', 'LOW'), 4))
        
        return exceptions
    
    def resolve_exception(self, exception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to resolve a planning exception.
        
        Args:
            exception: Exception dictionary with id and details
            
        Returns:
            Resolution result with status and actions taken
        """
        exception_id = exception.get('id')
        exception_type = exception.get('type')
        
        logger.info(
            "Attempting to resolve exception %s of type %s",
            exception_id,
            exception_type
        )
        
        resolution = {
            'exception_id': exception_id,
            'status': 'pending',
            'actions': [],
            'escalated': False
        }
        
        # Route to appropriate handler based on type
        if exception_type == 'stockout_risk':
            resolution = self._handle_stockout_exception(exception)
        elif exception_type == 'demand_anomaly':
            resolution = self._handle_demand_anomaly(exception)
        elif exception_type == 'supply_delay':
            resolution = self._handle_supply_delay(exception)
        else:
            resolution['status'] = 'escalated'
            resolution['escalated'] = True
            resolution['actions'].append('Escalated to manual review')
        
        return resolution
    
    def _analyze_demand_supply_gaps(
        self,
        demand_plan: pd.DataFrame,
        supply_plan: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Analyze gaps between demand and supply plans."""
        gaps = []
        
        # Simple gap analysis - in production this would be more sophisticated
        if not demand_plan.empty and not supply_plan.empty:
            # Placeholder logic
            gaps.append({
                'type': 'capacity_gap',
                'location': 'DC-001',
                'period': 'Week 4',
                'demand': 10000,
                'supply': 8000,
                'gap': 2000,
                'severity': 'HIGH'
            })
        
        return gaps
    
    def _evaluate_scenario(
        self,
        scenario: str,
        base_demand: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate a planning scenario."""
        multipliers = {
            'base': 1.0,
            'optimistic': 1.15,
            'pessimistic': 0.85
        }
        
        multiplier = multipliers.get(scenario, 1.0)
        
        return {
            'scenario': scenario,
            'demand_multiplier': multiplier,
            'total_demand': base_demand['quantity'].sum() * multiplier if not base_demand.empty else 0,
            'confidence': 0.9 if scenario == 'base' else 0.7
        }
    
    def _generate_recommendations(
        self,
        gaps: List[Dict[str, Any]],
        scenarios: Dict[str, Any]
    ) -> List[str]:
        """Generate planning recommendations based on analysis."""
        recommendations = []
        
        if gaps:
            recommendations.append(f"Address {len(gaps)} capacity gaps identified in the supply plan")
        
        if 'pessimistic' in scenarios:
            recommendations.append("Build safety stock buffer for pessimistic scenario coverage")
        
        recommendations.append("Review service level agreements for high-priority items")
        recommendations.append("Coordinate with suppliers on lead time improvements")
        
        return recommendations
    
    def _handle_stockout_exception(self, exception: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stockout risk exception."""
        return {
            'exception_id': exception.get('id'),
            'status': 'resolved',
            'actions': [
                'Generated emergency purchase order',
                'Initiated inter-store transfer',
                'Updated safety stock parameters'
            ],
            'escalated': False
        }
    
    def _handle_demand_anomaly(self, exception: Dict[str, Any]) -> Dict[str, Any]:
        """Handle demand anomaly exception."""
        return {
            'exception_id': exception.get('id'),
            'status': 'resolved',
            'actions': [
                'Adjusted short-term forecast',
                'Notified planning team',
                'Updated demand model'
            ],
            'escalated': False
        }
    
    def _handle_supply_delay(self, exception: Dict[str, Any]) -> Dict[str, Any]:
        """Handle supply delay exception."""
        return {
            'exception_id': exception.get('id'),
            'status': 'partially_resolved',
            'actions': [
                'Contacted alternate suppliers',
                'Adjusted allocation priorities',
                'Notified affected stores'
            ],
            'escalated': True
        }
