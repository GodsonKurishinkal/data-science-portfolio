#!/usr/bin/env python3
"""
Supply Chain Planning System - Interactive Demo

Demonstrates the unified planning capabilities across all modules.
"""

import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, '.')

from src.config.loader import PlanningConfig
from src.orchestrator.planner import SupplyChainPlanner
from src.orchestrator.workflow import PlanningWorkflow
from src.orchestrator.scheduler import PlanningScheduler
from src.utils.helpers import format_currency, format_percentage


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def demo_planning_cycle():
    """Demo 1: Complete Planning Cycle."""
    print_header("DEMO 1: Complete Planning Cycle")
    
    # Initialize
    config = PlanningConfig()
    planner = SupplyChainPlanner(config)
    
    print("\n✓ Initialized Supply Chain Planner")
    print("  - Modules: Demand, Inventory, Pricing, Network, Sensing, Replenishment")
    
    # Run planning cycle
    print_section("Running Monthly Planning Cycle")
    
    results = planner.run_planning_cycle(
        planning_horizon='monthly',
        include_modules=['demand', 'inventory', 'pricing', 'network', 'replenishment']
    )
    
    # Display results
    print_section("Planning Results")
    
    if results.demand:
        print("\n📈 DEMAND FORECASTING")
        print(f"   MAPE: {format_percentage(results.demand.mape)}")
        print(f"   Model: {results.demand.model_used}")
        print(f"   Elasticity: {results.demand.elasticity}")
    
    if results.inventory:
        print("\n📦 INVENTORY OPTIMIZATION")
        print(f"   Service Level: {format_percentage(results.inventory.service_level)}")
        print(f"   Inventory Value: {format_currency(results.inventory.total_inventory_value)}")
    
    if results.pricing:
        print("\n💰 DYNAMIC PRICING")
        print(f"   Revenue Lift: {format_percentage(results.pricing.revenue_lift)}")
        print(f"   Margin Improvement: {format_percentage(results.pricing.margin_improvement)}")
    
    if results.network:
        print("\n🚚 NETWORK OPTIMIZATION")
        print(f"   Cost Reduction: {format_percentage(results.network.cost_reduction)}")
        print(f"   Distance Savings: {format_percentage(results.network.distance_savings)}")
    
    if results.replenishment:
        print("\n🔄 AUTO-REPLENISHMENT")
        print(f"   Automation Rate: {format_percentage(results.replenishment.automation_rate)}")
        print(f"   Service Level: {format_percentage(results.replenishment.service_level_achieved)}")
    
    # KPIs
    print_section("Integrated KPIs")
    for kpi_name, value in results.kpis.items():
        if isinstance(value, float) and value < 10:
            print(f"   {kpi_name}: {format_percentage(value)}")
        else:
            print(f"   {kpi_name}: {value}")


def demo_sop_planning():
    """Demo 2: S&OP Planning."""
    print_header("DEMO 2: Sales & Operations Planning (S&OP)")
    
    config = PlanningConfig()
    planner = SupplyChainPlanner(config)
    
    print("\n✓ Generating 3-month S&OP plan with scenarios...")
    
    sop_plan = planner.generate_sop_plan(
        horizon_months=3,
        scenarios=['base', 'optimistic', 'pessimistic']
    )
    
    print_section("S&OP Plan Summary")
    print(f"   Planning Date: {sop_plan.planning_date}")
    print(f"   Horizon: {sop_plan.horizon_months} months")
    
    print_section("Scenario Analysis")
    for scenario, data in sop_plan.scenarios.items():
        print(f"\n   {scenario.upper()}")
        print(f"   - Demand Multiplier: {data['demand_multiplier']}")
        print(f"   - Confidence: {format_percentage(data['confidence'])}")
    
    print_section("Gaps Identified")
    for gap in sop_plan.gaps:
        print(f"   ⚠️ {gap['type']} at {gap['location']}")
        print(f"      Period: {gap['period']}, Gap: {gap['gap']} units")
    
    print_section("Recommendations")
    for i, rec in enumerate(sop_plan.recommendations, 1):
        print(f"   {i}. {rec}")


def demo_daily_replenishment():
    """Demo 3: Daily Replenishment."""
    print_header("DEMO 3: Daily Replenishment Run")
    
    config = PlanningConfig()
    planner = SupplyChainPlanner(config)
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n✓ Running daily replenishment for {today}")
    
    result = planner.run_daily_replenishment(
        run_date=today,
        scenarios=['dc_to_store', 'supplier_to_dc']
    )
    
    print_section("Replenishment Results")
    print(f"   Run Date: {result.run_date}")
    print(f"   Scenarios: {', '.join(result.scenarios)}")
    
    if not result.purchase_orders.empty:
        print(f"\n   📋 Purchase Orders Generated: {len(result.purchase_orders)}")
        print(result.purchase_orders.to_string(index=False))
    
    if not result.transfer_orders.empty:
        print(f"\n   📦 Transfer Orders Generated: {len(result.transfer_orders)}")
    
    print_section("Alerts")
    for alert in result.alerts:
        severity = alert.get('severity', 'INFO')
        msg = alert.get('message', alert.get('type', 'Alert'))
        print(f"   {'🔴' if severity == 'HIGH' else '🟡'} [{severity}] {msg}")


def demo_exception_handling():
    """Demo 4: Exception Handling."""
    print_header("DEMO 4: Exception Monitoring & Resolution")
    
    config = PlanningConfig()
    planner = SupplyChainPlanner(config)
    
    # Run sensing first to generate exceptions
    planner.run_planning_cycle(include_modules=['sensing', 'inventory', 'replenishment'])
    
    print("\n✓ Monitoring for exceptions across all modules...")
    
    exceptions = planner.monitor_exceptions()
    
    print_section(f"Exceptions Found: {len(exceptions)}")
    
    for exc in exceptions[:5]:  # Show first 5
        severity = exc.get('severity', 'UNKNOWN')
        exc_type = exc.get('type', 'unknown')
        item = exc.get('item_id', 'N/A')
        
        icon = '🔴' if severity == 'CRITICAL' else '🟠' if severity == 'HIGH' else '🟡'
        print(f"\n   {icon} [{severity}] {exc_type}")
        print(f"      Item: {item}")
        
        # Attempt resolution
        if severity in ['CRITICAL', 'HIGH']:
            resolution = planner.resolve_exception(exc)
            print(f"      Resolution: {resolution['status']}")
            for action in resolution.get('actions', [])[:2]:
                print(f"        → {action}")


def demo_workflow():
    """Demo 5: Planning Workflow."""
    print_header("DEMO 5: Planning Workflows")
    
    print("\n✓ Available workflow types:")
    print("   - monthly_sop: Monthly S&OP planning cycle")
    print("   - weekly: Weekly tactical planning")
    print("   - daily: Daily operational planning")
    print("   - realtime: Real-time exception handling")
    
    # Create daily workflow
    workflow = PlanningWorkflow('daily')
    
    print_section("Daily Workflow Steps")
    for step_name in workflow.execution_order:
        step = workflow.steps[step_name]
        deps = ', '.join(step.dependencies) if step.dependencies else 'None'
        print(f"   {step_name}")
        print(f"      Type: {step.step_type.value}")
        print(f"      Dependencies: {deps}")


def demo_scheduler():
    """Demo 6: Planning Scheduler."""
    print_header("DEMO 6: Planning Scheduler")
    
    scheduler = PlanningScheduler()
    
    print("\n✓ Scheduled Jobs:")
    
    summary = scheduler.get_schedule_summary()
    
    for job in summary['jobs']:
        print(f"\n   📅 {job['job_id']}")
        print(f"      Workflow: {job['workflow_type']}")
        print(f"      Frequency: {job['frequency']}")
        print(f"      Enabled: {'✓' if job['enabled'] else '✗'}")
        if job['next_run']:
            print(f"      Next Run: {job['next_run']}")


def main():
    """Main demo entry point."""
    print("\n" + "=" * 70)
    print("   🏭 SUPPLY CHAIN PLANNING SYSTEM - INTERACTIVE DEMO")
    print("   Master Orchestrator for End-to-End Planning")
    print("=" * 70)
    
    demos = {
        '1': ('Complete Planning Cycle', demo_planning_cycle),
        '2': ('S&OP Planning', demo_sop_planning),
        '3': ('Daily Replenishment', demo_daily_replenishment),
        '4': ('Exception Handling', demo_exception_handling),
        '5': ('Planning Workflows', demo_workflow),
        '6': ('Planning Scheduler', demo_scheduler),
        '0': ('Run All Demos', None),
    }
    
    print("\nAvailable Demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    
    print("\nEnter demo number (or 0 for all, q to quit):")
    
    while True:
        try:
            choice = input("\n> ").strip().lower()
            
            if choice == 'q':
                print("\nThank you for exploring the Supply Chain Planning System!")
                break
            
            if choice == '0':
                for key, (_, func) in demos.items():
                    if func:
                        func()
                break
            
            if choice in demos:
                name, func = demos[choice]
                if func:
                    func()
            else:
                print("Invalid choice. Enter 1-6, 0, or q.")
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted.")
            break
        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("   Demo Complete - Supply Chain Planning System")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
