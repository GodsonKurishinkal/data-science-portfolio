"""
Planning Scheduler.

Manages scheduling of planning workflows.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScheduleFrequency(Enum):
    """Schedule frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


@dataclass
class ScheduledJob:
    """A scheduled planning job."""
    
    job_id: str
    workflow_type: str
    frequency: ScheduleFrequency
    run_time: Optional[time] = None
    day_of_week: Optional[int] = None  # 0=Monday
    day_of_month: Optional[int] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class PlanningScheduler:
    """
    Manages scheduling of planning workflows.
    
    Supports:
    - Daily operational runs (replenishment, sensing)
    - Weekly tactical planning
    - Monthly S&OP cycles
    - On-demand execution
    
    Example:
        >>> scheduler = PlanningScheduler()
        >>> scheduler.add_job('daily_replenishment', 'daily', ScheduleFrequency.DAILY, time(6, 0))
        >>> scheduler.start()
    """
    
    def __init__(self):
        """Initialize the planning scheduler."""
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        self._setup_default_jobs()
        logger.info("PlanningScheduler initialized")
    
    def _setup_default_jobs(self) -> None:
        """Set up default scheduled jobs."""
        # Daily replenishment at 6 AM
        self.add_job(
            job_id='daily_replenishment',
            workflow_type='daily',
            frequency=ScheduleFrequency.DAILY,
            run_time=time(6, 0),
            config={'scenarios': ['dc_to_store', 'supplier_to_dc']}
        )
        
        # Daily pricing update at 7 AM
        self.add_job(
            job_id='daily_pricing',
            workflow_type='daily',
            frequency=ScheduleFrequency.DAILY,
            run_time=time(7, 0),
            config={'include_competitive': True}
        )
        
        # Weekly tactical planning on Monday at 9 AM
        self.add_job(
            job_id='weekly_tactical',
            workflow_type='weekly',
            frequency=ScheduleFrequency.WEEKLY,
            run_time=time(9, 0),
            day_of_week=0,  # Monday
            config={'horizon_weeks': 8}
        )
        
        # Monthly S&OP on 1st of month at 8 AM
        self.add_job(
            job_id='monthly_sop',
            workflow_type='monthly_sop',
            frequency=ScheduleFrequency.MONTHLY,
            run_time=time(8, 0),
            day_of_month=1,
            config={'horizon_months': 3, 'scenarios': ['base', 'optimistic', 'pessimistic']}
        )
    
    def add_job(
        self,
        job_id: str,
        workflow_type: str,
        frequency: ScheduleFrequency,
        run_time: Optional[time] = None,
        day_of_week: Optional[int] = None,
        day_of_month: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a scheduled job.
        
        Args:
            job_id: Unique job identifier
            workflow_type: Type of workflow to run
            frequency: How often to run
            run_time: Time of day to run
            day_of_week: Day of week for weekly jobs (0=Monday)
            day_of_month: Day of month for monthly jobs
            config: Job-specific configuration
        """
        job = ScheduledJob(
            job_id=job_id,
            workflow_type=workflow_type,
            frequency=frequency,
            run_time=run_time,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            config=config or {}
        )
        
        job.next_run = self._calculate_next_run(job)
        self.jobs[job_id] = job
        logger.info("Added job %s with frequency %s", job_id, frequency.value)
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info("Removed job %s", job_id)
            return True
        return False
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a job."""
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
            return True
        return False
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a job."""
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
            return True
        return False
    
    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[ScheduledJob]:
        """List all scheduled jobs."""
        return list(self.jobs.values())
    
    def get_pending_jobs(self) -> List[ScheduledJob]:
        """Get jobs that are due to run."""
        now = datetime.now()
        pending = []
        
        for job in self.jobs.values():
            if job.enabled and job.next_run and job.next_run <= now:
                pending.append(job)
        
        return sorted(pending, key=lambda j: j.next_run or now)
    
    def _calculate_next_run(self, job: ScheduledJob) -> Optional[datetime]:
        """Calculate the next run time for a job."""
        now = datetime.now()
        
        if job.frequency == ScheduleFrequency.ON_DEMAND:
            return None
        
        if job.frequency == ScheduleFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run.replace(hour=next_run.hour + 1)
            return next_run
        
        if job.frequency == ScheduleFrequency.DAILY:
            if job.run_time:
                next_run = now.replace(
                    hour=job.run_time.hour,
                    minute=job.run_time.minute,
                    second=0,
                    microsecond=0
                )
                if next_run <= now:
                    next_run = next_run.replace(day=next_run.day + 1)
                return next_run
        
        if job.frequency == ScheduleFrequency.WEEKLY:
            if job.run_time and job.day_of_week is not None:
                days_ahead = job.day_of_week - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                next_run = now.replace(
                    hour=job.run_time.hour,
                    minute=job.run_time.minute,
                    second=0,
                    microsecond=0
                )
                from datetime import timedelta
                next_run = next_run + timedelta(days=days_ahead)
                return next_run
        
        if job.frequency == ScheduleFrequency.MONTHLY:
            if job.run_time and job.day_of_month:
                next_run = now.replace(
                    day=job.day_of_month,
                    hour=job.run_time.hour,
                    minute=job.run_time.minute,
                    second=0,
                    microsecond=0
                )
                if next_run <= now:
                    # Move to next month
                    if now.month == 12:
                        next_run = next_run.replace(year=now.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=now.month + 1)
                return next_run
        
        return None
    
    def run_job(self, job_id: str, planner: Any) -> Dict[str, Any]:
        """
        Run a specific job immediately.
        
        Args:
            job_id: Job to run
            planner: SupplyChainPlanner instance
            
        Returns:
            Job execution result
        """
        job = self.jobs.get(job_id)
        if not job:
            return {'status': 'error', 'message': f'Job {job_id} not found'}
        
        logger.info("Running job %s", job_id)
        
        try:
            from src.orchestrator.workflow import PlanningWorkflow
            
            workflow = PlanningWorkflow(job.workflow_type)
            result = workflow.execute(planner, job.config)
            
            job.last_run = datetime.now()
            job.next_run = self._calculate_next_run(job)
            
            return {
                'status': 'success',
                'job_id': job_id,
                'workflow_result': result
            }
            
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, str(e))
            return {
                'status': 'error',
                'job_id': job_id,
                'message': str(e)
            }
    
    def start(self) -> None:
        """Start the scheduler (placeholder for production implementation)."""
        self.running = True
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        logger.info("Scheduler stopped")
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get a summary of all scheduled jobs."""
        return {
            'total_jobs': len(self.jobs),
            'enabled_jobs': sum(1 for j in self.jobs.values() if j.enabled),
            'jobs': [
                {
                    'job_id': j.job_id,
                    'workflow_type': j.workflow_type,
                    'frequency': j.frequency.value,
                    'enabled': j.enabled,
                    'next_run': j.next_run.isoformat() if j.next_run else None,
                    'last_run': j.last_run.isoformat() if j.last_run else None
                }
                for j in self.jobs.values()
            ]
        }
