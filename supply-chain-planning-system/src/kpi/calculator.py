"""
KPI Calculator for Supply Chain Planning System.
"""

from typing import Dict, Any, Optional
import logging

from src.data.models import PlanningResult

logger = logging.getLogger(__name__)


class KPICalculator:
    """
    Calculates integrated KPIs across all planning modules.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.kpi_definitions = self._load_kpi_definitions()
        logger.info("KPICalculator initialized")
    
    def _load_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load KPI definitions."""
        return {
            'forecast_accuracy': {
                'name': 'Forecast Accuracy (MAPE)',
                'target': 0.15,
                'direction': 'lower_is_better',
                'unit': 'percentage'
            },
            'service_level': {
                'name': 'Service Level',
                'target': 0.98,
                'direction': 'higher_is_better',
                'unit': 'percentage'
            },
            'inventory_turns': {
                'name': 'Inventory Turns',
                'target': 12.0,
                'direction': 'higher_is_better',
                'unit': 'ratio'
            },
            'revenue_lift': {
                'name': 'Revenue Lift',
                'target': 0.08,
                'direction': 'higher_is_better',
                'unit': 'percentage'
            },
            'logistics_cost_reduction': {
                'name': 'Logistics Cost Reduction',
                'target': 0.15,
                'direction': 'higher_is_better',
                'unit': 'percentage'
            },
            'automation_rate': {
                'name': 'Replenishment Automation Rate',
                'target': 0.80,
                'direction': 'higher_is_better',
                'unit': 'percentage'
            }
        }
    
    def calculate_all(self, results: PlanningResult) -> Dict[str, float]:
        """
        Calculate all KPIs from planning results.
        
        Args:
            results: PlanningResult from planning cycle
            
        Returns:
            Dictionary of KPI name to value
        """
        kpis = {}
        
        # Demand KPIs
        if results.demand:
            kpis['forecast_accuracy'] = 1 - results.demand.mape
            kpis['forecast_bias'] = results.demand.bias
        
        # Inventory KPIs
        if results.inventory:
            kpis['service_level'] = results.inventory.service_level
            kpis['inventory_value'] = results.inventory.total_inventory_value
        
        # Pricing KPIs
        if results.pricing:
            kpis['revenue_lift'] = results.pricing.revenue_lift
            kpis['margin_improvement'] = results.pricing.margin_improvement
        
        # Network KPIs
        if results.network:
            kpis['logistics_cost_reduction'] = results.network.cost_reduction
            kpis['route_efficiency'] = 1 - results.network.distance_savings
        
        # Replenishment KPIs
        if results.replenishment:
            kpis['automation_rate'] = results.replenishment.automation_rate
            kpis['replenishment_service_level'] = results.replenishment.service_level_achieved
        
        return kpis
    
    def evaluate_against_targets(self, kpis: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate KPIs against targets.
        
        Args:
            kpis: Calculated KPI values
            
        Returns:
            KPI evaluation results with status
        """
        evaluation = {}
        
        for kpi_name, value in kpis.items():
            definition = self.kpi_definitions.get(kpi_name)
            
            if definition:
                target = definition['target']
                direction = definition['direction']
                
                if direction == 'higher_is_better':
                    status = 'green' if value >= target else 'red'
                    variance = value - target
                else:
                    status = 'green' if value <= target else 'red'
                    variance = target - value
                
                evaluation[kpi_name] = {
                    'value': value,
                    'target': target,
                    'status': status,
                    'variance': variance,
                    'name': definition['name']
                }
            else:
                evaluation[kpi_name] = {
                    'value': value,
                    'target': None,
                    'status': 'unknown',
                    'variance': None,
                    'name': kpi_name
                }
        
        return evaluation
    
    def get_summary(self, kpis: Dict[str, float]) -> Dict[str, Any]:
        """Get a summary of KPI performance."""
        evaluation = self.evaluate_against_targets(kpis)
        
        green_count = sum(1 for e in evaluation.values() if e['status'] == 'green')
        red_count = sum(1 for e in evaluation.values() if e['status'] == 'red')
        
        return {
            'total_kpis': len(kpis),
            'on_target': green_count,
            'below_target': red_count,
            'health_score': green_count / len(kpis) if kpis else 0,
            'kpis': evaluation
        }
