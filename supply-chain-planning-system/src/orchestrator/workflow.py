"""
Planning Workflow Definitions.

Defines the workflow patterns for different planning cycles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    DEMAND = "demand"
    INVENTORY = "inventory"
    PRICING = "pricing"
    NETWORK = "network"
    SENSING = "sensing"
    REPLENISHMENT = "replenishment"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"


@dataclass
class WorkflowStep:
    """A single step in a planning workflow."""
    
    name: str
    step_type: WorkflowStepType
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    steps: Dict[str, WorkflowStep]
    output: Any = None
    errors: List[str] = field(default_factory=list)


class PlanningWorkflow:
    """
    Manages planning workflow definitions and execution.
    
    Supports different planning cycles:
    - Monthly S&OP
    - Weekly tactical planning
    - Daily operational planning
    - Real-time exception handling
    
    Example:
        >>> workflow = PlanningWorkflow('monthly_sop')
        >>> workflow.add_step('demand', WorkflowStepType.DEMAND, dependencies=[])
        >>> workflow.add_step('inventory', WorkflowStepType.INVENTORY, dependencies=['demand'])
        >>> result = workflow.execute(planner)
    """
    
    def __init__(self, workflow_type: str):
        """
        Initialize a planning workflow.
        
        Args:
            workflow_type: Type of workflow ('monthly_sop', 'weekly', 'daily', 'realtime')
        """
        self.workflow_type = workflow_type
        self.workflow_id = f"{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []
        self._setup_default_workflow()
        logger.info(f"PlanningWorkflow initialized: {self.workflow_id}")
    
    def _setup_default_workflow(self) -> None:
        """Set up default workflow based on type."""
        if self.workflow_type == 'monthly_sop':
            self._setup_monthly_sop_workflow()
        elif self.workflow_type == 'weekly':
            self._setup_weekly_workflow()
        elif self.workflow_type == 'daily':
            self._setup_daily_workflow()
        elif self.workflow_type == 'realtime':
            self._setup_realtime_workflow()
    
    def _setup_monthly_sop_workflow(self) -> None:
        """Set up monthly S&OP workflow."""
        self.add_step('demand_forecast', WorkflowStepType.DEMAND, 
                      dependencies=[], config={'horizon_months': 3})
        self.add_step('inventory_optimization', WorkflowStepType.INVENTORY,
                      dependencies=['demand_forecast'])
        self.add_step('pricing_optimization', WorkflowStepType.PRICING,
                      dependencies=['demand_forecast'])
        self.add_step('network_optimization', WorkflowStepType.NETWORK,
                      dependencies=['inventory_optimization'])
        self.add_step('aggregation', WorkflowStepType.AGGREGATION,
                      dependencies=['inventory_optimization', 'pricing_optimization', 'network_optimization'])
        self.add_step('validation', WorkflowStepType.VALIDATION,
                      dependencies=['aggregation'])
    
    def _setup_weekly_workflow(self) -> None:
        """Set up weekly tactical planning workflow."""
        self.add_step('demand_update', WorkflowStepType.DEMAND,
                      dependencies=[], config={'horizon_weeks': 8})
        self.add_step('inventory_review', WorkflowStepType.INVENTORY,
                      dependencies=['demand_update'])
        self.add_step('replenishment_planning', WorkflowStepType.REPLENISHMENT,
                      dependencies=['inventory_review'])
    
    def _setup_daily_workflow(self) -> None:
        """Set up daily operational workflow."""
        self.add_step('sensing', WorkflowStepType.SENSING, dependencies=[])
        self.add_step('replenishment', WorkflowStepType.REPLENISHMENT,
                      dependencies=['sensing'])
        self.add_step('pricing_update', WorkflowStepType.PRICING,
                      dependencies=['sensing'])
    
    def _setup_realtime_workflow(self) -> None:
        """Set up real-time exception handling workflow."""
        self.add_step('sensing', WorkflowStepType.SENSING, dependencies=[])
        self.add_step('exception_handling', WorkflowStepType.REPLENISHMENT,
                      dependencies=['sensing'], config={'mode': 'exception_only'})
    
    def add_step(
        self,
        name: str,
        step_type: WorkflowStepType,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
        retry_count: int = 3
    ) -> None:
        """
        Add a step to the workflow.
        
        Args:
            name: Unique step name
            step_type: Type of step
            dependencies: List of step names this step depends on
            config: Step-specific configuration
            timeout_seconds: Step timeout
            retry_count: Number of retries on failure
        """
        if dependencies is None:
            dependencies = []
        if config is None:
            config = {}
        
        step = WorkflowStep(
            name=name,
            step_type=step_type,
            dependencies=dependencies,
            config=config,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count
        )
        
        self.steps[name] = step
        self._update_execution_order()
        logger.debug(f"Added workflow step: {name}")
    
    def _update_execution_order(self) -> None:
        """Update the execution order based on dependencies (topological sort)."""
        visited = set()
        order = []
        
        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            step = self.steps.get(name)
            if step:
                for dep in step.dependencies:
                    visit(dep)
                order.append(name)
        
        for step_name in self.steps:
            visit(step_name)
        
        self.execution_order = order
    
    def execute(self, planner: Any, data: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute the workflow.
        
        Args:
            planner: SupplyChainPlanner instance
            data: Optional input data
            
        Returns:
            WorkflowResult with execution details
        """
        started_at = datetime.now()
        errors = []
        
        logger.info(f"Executing workflow {self.workflow_id}")
        
        # Execute steps in order
        step_results = {}
        for step_name in self.execution_order:
            step = self.steps[step_name]
            step.status = WorkflowStatus.RUNNING
            
            try:
                logger.info(f"Executing step: {step_name}")
                
                # Get dependency results
                dep_results = {dep: step_results.get(dep) for dep in step.dependencies}
                
                # Execute step based on type
                result = self._execute_step(planner, step, dep_results, data)
                
                step.result = result
                step.status = WorkflowStatus.COMPLETED
                step_results[step_name] = result
                
            except Exception as e:
                error_msg = f"Step {step_name} failed: {str(e)}"
                logger.error(error_msg)
                step.status = WorkflowStatus.FAILED
                step.error = str(e)
                errors.append(error_msg)
                
                # Continue or fail based on step criticality
                if step.step_type in [WorkflowStepType.DEMAND, WorkflowStepType.INVENTORY]:
                    break  # Critical step failed
        
        # Determine overall status
        if errors:
            status = WorkflowStatus.FAILED
        elif all(s.status == WorkflowStatus.COMPLETED for s in self.steps.values()):
            status = WorkflowStatus.COMPLETED
        else:
            status = WorkflowStatus.FAILED
        
        return WorkflowResult(
            workflow_id=self.workflow_id,
            status=status,
            started_at=started_at,
            completed_at=datetime.now(),
            steps=self.steps,
            output=step_results,
            errors=errors
        )
    
    def _execute_step(
        self,
        planner: Any,
        step: WorkflowStep,
        dep_results: Dict[str, Any],
        data: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a single workflow step."""
        step_type = step.step_type
        config = step.config
        
        if step_type == WorkflowStepType.DEMAND:
            return planner.demand.run(data.get('demand') if data else None, **config)
        elif step_type == WorkflowStepType.INVENTORY:
            forecast = dep_results.get('demand_forecast', {})
            return planner.inventory.run(data.get('inventory') if data else None, forecast=forecast)
        elif step_type == WorkflowStepType.PRICING:
            return planner.pricing.run(data.get('pricing') if data else None)
        elif step_type == WorkflowStepType.NETWORK:
            return planner.network.run(data.get('network') if data else None)
        elif step_type == WorkflowStepType.SENSING:
            return planner.sensing.run(data.get('sensing') if data else None)
        elif step_type == WorkflowStepType.REPLENISHMENT:
            return planner.replenishment.run(data.get('replenishment') if data else None)
        elif step_type == WorkflowStepType.AGGREGATION:
            return {'aggregated': dep_results}
        elif step_type == WorkflowStepType.VALIDATION:
            return {'validated': True, 'results': dep_results}
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    def get_status(self) -> Dict[str, WorkflowStatus]:
        """Get status of all steps."""
        return {name: step.status for name, step in self.steps.items()}
    
    def reset(self) -> None:
        """Reset workflow for re-execution."""
        for step in self.steps.values():
            step.status = WorkflowStatus.PENDING
            step.result = None
            step.error = None
        logger.info(f"Workflow {self.workflow_id} reset")
