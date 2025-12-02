"""
Tests for Supply Chain Planning System orchestrator module.
"""

from datetime import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import PlanningConfig
from src.orchestrator.planner import SupplyChainPlanner
from src.orchestrator.workflow import PlanningWorkflow, WorkflowStatus, WorkflowStepType
from src.orchestrator.scheduler import PlanningScheduler, ScheduleFrequency


class TestSupplyChainPlanner:
    """Tests for SupplyChainPlanner class."""

    def test_initialization(self):
        """Test planner initialization."""
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        assert planner.config is not None
        assert planner.demand is not None
        assert planner.inventory is not None

    def test_run_planning_cycle(self):
        """Test running a complete planning cycle."""
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        results = planner.run_planning_cycle(
            include_modules=['demand', 'inventory']
        )

        assert results is not None
        assert results.demand is not None
        assert results.inventory is not None

    def test_planning_cycle_with_all_modules(self):
        """Test planning cycle with all modules."""
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        results = planner.run_planning_cycle()

        assert results is not None
        assert results.demand is not None
        assert results.inventory is not None
        assert results.pricing is not None
        assert results.network is not None
        assert results.sensing is not None
        assert results.replenishment is not None

    def test_run_daily_replenishment(self):
        """Test daily replenishment run."""
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        result = planner.run_daily_replenishment(run_date='2024-01-01')

        assert result is not None
        assert result.purchase_orders is not None or result.transfer_orders is not None

    def test_monitor_exceptions(self):
        """Test exception monitoring."""
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        # Run planning first to generate state
        planner.run_planning_cycle(include_modules=['sensing', 'inventory'])

        exceptions = planner.monitor_exceptions()

        assert isinstance(exceptions, list)


class TestPlanningWorkflow:
    """Tests for PlanningWorkflow class."""

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = PlanningWorkflow('daily')

        assert workflow is not None
        assert workflow.workflow_type == 'daily'
        assert len(workflow.steps) > 0

    def test_default_workflows(self):
        """Test all default workflow types."""
        workflow_types = ['monthly_sop', 'weekly', 'daily', 'realtime']

        for wf_type in workflow_types:
            workflow = PlanningWorkflow(wf_type)
            assert workflow.workflow_type == wf_type
            assert len(workflow.steps) > 0

    def test_execution_order(self):
        """Test workflow execution order."""
        workflow = PlanningWorkflow('daily')

        assert len(workflow.execution_order) > 0
        # Execution order should respect dependencies
        executed = set()
        for step_name in workflow.execution_order:
            step = workflow.steps[step_name]
            # All dependencies should have been executed
            for dep in step.dependencies:
                assert dep in executed or dep not in workflow.steps
            executed.add(step_name)

    def test_add_step(self):
        """Test adding a step to workflow."""
        workflow = PlanningWorkflow('daily')
        initial_count = len(workflow.steps)

        workflow.add_step(
            name='custom_step',
            step_type=WorkflowStepType.AGGREGATION,
            dependencies=['sensing']
        )

        assert len(workflow.steps) == initial_count + 1
        assert 'custom_step' in workflow.steps

    def test_workflow_execution(self):
        """Test workflow execution."""
        workflow = PlanningWorkflow('daily')
        config = PlanningConfig()
        planner = SupplyChainPlanner(config)

        result = workflow.execute(planner)

        assert result is not None
        assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
        assert result.completed_at is not None

    def test_get_status(self):
        """Test getting workflow status."""
        workflow = PlanningWorkflow('daily')

        status = workflow.get_status()

        assert status is not None
        assert isinstance(status, dict)

    def test_reset(self):
        """Test workflow reset."""
        workflow = PlanningWorkflow('daily')

        workflow.reset()

        for step in workflow.steps.values():
            assert step.status == WorkflowStatus.PENDING


class TestPlanningScheduler:
    """Tests for PlanningScheduler class."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = PlanningScheduler()

        assert scheduler is not None
        assert len(scheduler.jobs) > 0

    def test_add_job(self):
        """Test adding a job."""
        scheduler = PlanningScheduler()
        initial_count = len(scheduler.jobs)

        scheduler.add_job(
            job_id='test_job',
            workflow_type='daily',
            frequency=ScheduleFrequency.DAILY,
            run_time=time(10, 0)
        )

        assert len(scheduler.jobs) == initial_count + 1
        assert 'test_job' in scheduler.jobs

    def test_remove_job(self):
        """Test removing a job."""
        scheduler = PlanningScheduler()
        scheduler.add_job('temp_job', 'daily', ScheduleFrequency.DAILY, time(10, 0))

        result = scheduler.remove_job('temp_job')

        assert result is True
        assert 'temp_job' not in scheduler.jobs

    def test_enable_disable_job(self):
        """Test enabling/disabling jobs."""
        scheduler = PlanningScheduler()
        scheduler.add_job('toggle_job', 'daily', ScheduleFrequency.DAILY, time(10, 0))

        scheduler.disable_job('toggle_job')
        assert not scheduler.jobs['toggle_job'].enabled

        scheduler.enable_job('toggle_job')
        assert scheduler.jobs['toggle_job'].enabled

    def test_get_schedule_summary(self):
        """Test getting schedule summary."""
        scheduler = PlanningScheduler()

        summary = scheduler.get_schedule_summary()

        assert 'jobs' in summary
        assert len(summary['jobs']) > 0

        for job in summary['jobs']:
            assert 'job_id' in job
            assert 'workflow_type' in job
            assert 'frequency' in job

    def test_get_pending_jobs(self):
        """Test getting pending jobs."""
        scheduler = PlanningScheduler()

        pending_jobs = scheduler.get_pending_jobs()

        # Should return a list (may be empty if no jobs are due)
        assert isinstance(pending_jobs, list)

    def test_list_jobs(self):
        """Test listing all jobs."""
        scheduler = PlanningScheduler()

        jobs = scheduler.list_jobs()

        assert isinstance(jobs, list)
        assert len(jobs) > 0
