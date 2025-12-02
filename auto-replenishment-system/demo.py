#!/usr/bin/env python
"""Universal Replenishment Engine - Interactive Demo.

This demo showcases the replenishment engine's capabilities across
multiple retail scenarios without requiring actual data files.

Run: python demo.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.engine.replenishment import ReplenishmentEngine, ReplenishmentResult
from src.classification.abc_classifier import ABCClassifier
from src.classification.xyz_classifier import XYZClassifier
from src.classification.matrix import ABCXYZMatrix
from src.policies.periodic_review import PeriodicReviewPolicy
from src.safety_stock.calculator import SafetyStockCalculator
from src.alerts.generator import AlertGenerator, AlertThresholds
from src.alerts.types import AlertSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_inventory(n_items: int = 100) -> pd.DataFrame:
    """Generate sample inventory data."""
    np.random.seed(42)
    
    return pd.DataFrame({
        "item_id": [f"SKU{i:05d}" for i in range(n_items)],
        "item_name": [f"Product {i}" for i in range(n_items)],
        "current_stock": np.random.randint(0, 500, n_items),
        "max_capacity": np.random.randint(500, 1000, n_items),
        "daily_demand_rate": np.random.uniform(5, 80, n_items),
        "demand_std": np.random.uniform(2, 30, n_items),
        "unit_cost": np.random.uniform(5, 200, n_items),
        "revenue": np.random.uniform(1000, 500000, n_items),
        "lead_time": np.random.choice([3, 7, 14, 21], n_items),
        "category": np.random.choice(
            ["Electronics", "Apparel", "Grocery", "Home"], n_items
        ),
    })


def generate_sample_demand(
    n_items: int = 100, 
    n_days: int = 90
) -> pd.DataFrame:
    """Generate sample demand history."""
    np.random.seed(42)
    
    items = [f"SKU{i:05d}" for i in range(n_items)]
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_days,
        freq="D",
    )
    
    data = []
    for item in items:
        base_demand = np.random.uniform(10, 100)
        for date in dates:
            # Add day-of-week pattern
            dow_factor = 1.2 if date.dayofweek >= 5 else 1.0
            # Add noise
            noise = np.random.normal(0, base_demand * 0.3)
            demand = max(0, base_demand * dow_factor + noise)
            
            data.append({
                "item_id": item,
                "date": date,
                "quantity": demand,
            })
    
    return pd.DataFrame(data)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 60
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n--- {title} ---\n")


def demo_classification():
    """Demonstrate ABC/XYZ classification."""
    print_header("ABC/XYZ CLASSIFICATION DEMO")
    
    # Generate sample data
    inventory = generate_sample_inventory(50)
    demand = generate_sample_demand(50, 30)
    
    # ABC Classification
    print_subheader("ABC Classification (by Revenue)")
    
    abc_classifier = ABCClassifier(
        value_column="revenue",
        thresholds=(0.80, 0.95),
    )
    
    abc_result = abc_classifier.classify(inventory)
    
    # Show class distribution
    class_dist = abc_result["abc_class"].value_counts().sort_index()
    print("Class Distribution:")
    for cls, count in class_dist.items():
        pct = count / len(abc_result) * 100
        value = inventory.loc[abc_result[abc_result["abc_class"] == cls].index, "revenue"].sum()
        value_pct = value / inventory["revenue"].sum() * 100
        print(f"  Class {cls}: {count:3d} items ({pct:5.1f}%) - ${value:,.0f} ({value_pct:5.1f}% of value)")
    
    # XYZ Classification
    print_subheader("XYZ Classification (by Demand Variability)")
    
    xyz_classifier = XYZClassifier(cv_thresholds=(0.5, 1.0))
    xyz_result = xyz_classifier.classify(demand)
    
    xyz_dist = xyz_result["xyz_class"].value_counts().sort_index()
    print("Class Distribution:")
    for cls, count in xyz_dist.items():
        pct = count / len(xyz_result) * 100
        avg_cv = xyz_result[xyz_result["xyz_class"] == cls]["cv"].mean()
        print(f"  Class {cls}: {count:3d} items ({pct:5.1f}%) - Avg CV: {avg_cv:.2f}")
    
    # ABC-XYZ Matrix
    print_subheader("ABC-XYZ Service Level Matrix")
    
    matrix = ABCXYZMatrix()
    matrix_df = matrix.to_dataframe()
    
    # Display as a grid
    print("Service Levels by Classification:")
    print("        X      Y      Z")
    for abc in ["A", "B", "C"]:
        row = []
        for xyz in ["X", "Y", "Z"]:
            sl = matrix.get_service_level(abc, xyz)
            row.append(f"{sl:.0%}")
        print(f"  {abc}:   {'   '.join(row)}")
    
    return abc_result, xyz_result


def demo_policies():
    """Demonstrate replenishment policies."""
    print_header("REPLENISHMENT POLICY DEMO")
    
    # Generate test data with known characteristics
    test_data = pd.DataFrame({
        "item_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "item_name": ["Critical Item", "Low Stock", "Normal", "Overstocked", "Out of Stock"],
        "current_stock": [50, 30, 200, 800, 0],
        "max_capacity": [500, 500, 500, 500, 500],
        "daily_demand_rate": [20, 15, 10, 5, 25],
        "demand_std": [5, 4, 3, 2, 8],
        "unit_cost": [50, 30, 20, 15, 100],
        "inventory_position": [50, 30, 200, 800, 0],
    })
    
    # Periodic Review Policy
    print_subheader("Periodic Review (s,S) Policy")
    
    policy = PeriodicReviewPolicy(
        review_period=7,
        lead_time=7,
        service_level=0.95,
        order_strategy="policy_target",
    )
    
    result = policy.calculate(test_data)
    
    print(f"Policy Parameters:")
    print(f"  Review Period: {policy.review_period} days")
    print(f"  Lead Time: {policy.lead_time} days")
    print(f"  Service Level: {policy.service_level:.0%}")
    print()
    
    print("Replenishment Recommendations:")
    print("-" * 90)
    print(f"{'Item':<15} {'Stock':>8} {'DOS':>6} {'s':>8} {'S':>8} {'SS':>8} {'Order?':>8} {'Qty':>8}")
    print("-" * 90)
    
    for _, row in result.iterrows():
        dos = row["current_stock"] / row["daily_demand_rate"] if row["daily_demand_rate"] > 0 else float("inf")
        needs = "YES" if row["needs_order"] else "no"
        qty = int(row["recommended_quantity"]) if row["recommended_quantity"] > 0 else "-"
        
        print(
            f"{row['item_name']:<15} "
            f"{int(row['current_stock']):>8} "
            f"{dos:>6.1f} "
            f"{int(row['reorder_point']):>8} "
            f"{int(row['order_up_to_level']):>8} "
            f"{int(row['safety_stock']):>8} "
            f"{needs:>8} "
            f"{qty:>8}"
        )
    
    print("-" * 90)
    print(f"\nTotal Items Needing Order: {result['needs_order'].sum()}")
    print(f"Total Recommended Quantity: {result['recommended_quantity'].sum():,.0f} units")
    
    return result


def demo_safety_stock():
    """Demonstrate safety stock calculations."""
    print_header("SAFETY STOCK CALCULATION DEMO")
    
    # Test different scenarios
    scenarios = [
        {"name": "Low Variability (X)", "demand_mean": 100, "demand_std": 10, "lead_time": 7},
        {"name": "Medium Variability (Y)", "demand_mean": 100, "demand_std": 50, "lead_time": 7},
        {"name": "High Variability (Z)", "demand_mean": 100, "demand_std": 100, "lead_time": 7},
        {"name": "Long Lead Time", "demand_mean": 100, "demand_std": 30, "lead_time": 21},
    ]
    
    calculator = SafetyStockCalculator(method="standard")
    
    print("Safety Stock by Scenario (95% Service Level):")
    print("-" * 65)
    print(f"{'Scenario':<25} {'DDR':>8} {'σ':>8} {'LT':>6} {'SS':>10}")
    print("-" * 65)
    
    for scenario in scenarios:
        ss = calculator.calculate(
            demand_mean=scenario["demand_mean"],
            demand_std=scenario["demand_std"],
            lead_time=scenario["lead_time"],
            service_level=0.95,
        )
        print(
            f"{scenario['name']:<25} "
            f"{scenario['demand_mean']:>8.0f} "
            f"{scenario['demand_std']:>8.0f} "
            f"{scenario['lead_time']:>6} "
            f"{ss:>10.0f}"
        )
    
    # Compare service levels
    print_subheader("Impact of Service Level on Safety Stock")
    
    service_levels = [0.90, 0.95, 0.97, 0.99]
    base_case = {"demand_mean": 100, "demand_std": 30, "lead_time": 7}
    
    print(f"Base Case: DDR=100, σ=30, LT=7 days")
    print("-" * 40)
    print(f"{'Service Level':>15} {'Z-Score':>10} {'SS':>10}")
    print("-" * 40)
    
    for sl in service_levels:
        z = calculator._get_z_score(sl)
        ss = calculator.calculate(
            service_level=sl,
            **base_case,
        )
        print(f"{sl:>14.0%} {z:>10.2f} {ss:>10.0f}")


def demo_alerts():
    """Demonstrate alert generation."""
    print_header("ALERT GENERATION DEMO")
    
    # Create test data with various alert conditions
    test_data = pd.DataFrame({
        "item_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005", "SKU006"],
        "item_name": ["Stockout", "Critical", "Low Stock", "Overstock", "Capacity", "Healthy"],
        "current_stock": [0, 5, 20, 1000, 950, 150],
        "max_capacity": [500, 500, 500, 500, 1000, 500],
        "daily_demand_rate": [10, 10, 10, 10, 10, 10],
        "inventory_position": [0, 5, 20, 1000, 950, 150],
        "recommended_quantity": [100, 80, 50, 0, 0, 0],
        "source_available": [50, 100, 100, 100, 100, 100],  # SKU001 has shortage
        "reorder_point": [70, 70, 70, 70, 70, 70],
    })
    
    generator = AlertGenerator(
        thresholds=AlertThresholds(
            critical_days_supply=1.0,
            low_days_supply=3.0,
            overstock_days_supply=60.0,
            capacity_utilization_critical=0.95,
        )
    )
    
    alerts = generator.generate(test_data)
    
    print(f"Generated {len(alerts)} alerts from {len(test_data)} items\n")
    
    # Group by severity
    severity_order = [
        AlertSeverity.CRITICAL,
        AlertSeverity.HIGH,
        AlertSeverity.MEDIUM,
        AlertSeverity.LOW,
    ]
    
    for severity in severity_order:
        severity_alerts = [a for a in alerts if a.severity == severity]
        if severity_alerts:
            print(f"\n{severity.value.upper()} ALERTS ({len(severity_alerts)}):")
            print("-" * 70)
            for alert in severity_alerts:
                print(f"  [{alert.alert_type.value}] {alert.item_id}: {alert.message}")
                print(f"    → Action: {alert.recommended_action}")
    
    # Summary
    print_subheader("Alert Summary")
    summary = generator.summarize_alerts(alerts)
    
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Items Affected: {summary['items_affected']}")
    print("\nBy Severity:")
    for sev, count in summary["by_severity"].items():
        print(f"  {sev}: {count}")


def demo_full_engine():
    """Demonstrate the full replenishment engine."""
    print_header("FULL ENGINE DEMO - DC TO STORE SCENARIO")
    
    # Create sample data
    inventory = generate_sample_inventory(100)
    demand = generate_sample_demand(100, 60)
    
    # Create engine with configuration
    config = {
        "scenarios": {
            "dc_to_store": {
                "policy_type": "periodic_review",
                "review_period": 1,
                "lead_time": 2,
                "service_level": 0.97,
                "order_strategy": "fill_to_capacity",
            }
        },
        "classification": {
            "abc_enabled": True,
            "xyz_enabled": True,
        },
        "service_level_matrix": {
            "AX": 0.99, "AY": 0.97, "AZ": 0.95,
            "BX": 0.97, "BY": 0.95, "BZ": 0.92,
            "CX": 0.95, "CY": 0.92, "CZ": 0.90,
        },
        "alerts": {
            "enabled": True,
            "critical_days_supply": 1.0,
            "low_days_supply": 3.0,
        },
    }
    
    engine = ReplenishmentEngine(config_dict=config)
    
    print("Running replenishment calculation...")
    print(f"  Scenario: DC to Store")
    print(f"  Items: {len(inventory)}")
    print(f"  Demand History: {len(demand)} records")
    
    result = engine.run(
        scenario="dc_to_store",
        inventory_data=inventory,
        demand_data=demand,
    )
    
    # Summary
    print_subheader("Execution Summary")
    
    print(f"Total Items Analyzed: {result.summary['total_items']}")
    print(f"Items Needing Order: {result.summary['items_needing_order']}")
    print(f"Items Not Needing Order: {result.summary['items_not_needing_order']}")
    print(f"Total Recommended Quantity: {result.summary['total_recommended_quantity']:,.0f}")
    print(f"Total Alerts Generated: {result.summary['total_alerts']}")
    print(f"  - Critical: {result.summary.get('critical_alerts', 0)}")
    print(f"  - High: {result.summary.get('high_alerts', 0)}")
    
    # Top priority orders
    print_subheader("Top 10 Priority Orders")
    
    priority = result.get_priority_orders(top_n=10)
    
    if not priority.empty:
        print(f"{'Item':<12} {'Stock':>8} {'DOS':>6} {'Order Qty':>10} {'Category':<12}")
        print("-" * 55)
        
        for _, row in priority.iterrows():
            dos = row.get("days_of_supply", 0)
            print(
                f"{row['item_id']:<12} "
                f"{int(row.get('current_stock', 0)):>8} "
                f"{dos:>6.1f} "
                f"{int(row.get('recommended_quantity', 0)):>10} "
                f"{row.get('category', 'N/A'):<12}"
            )
    
    # Critical alerts
    if result.alerts:
        print_subheader("Critical & High Priority Alerts")
        
        critical_high = [
            a for a in result.alerts 
            if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
        ][:5]
        
        for alert in critical_high:
            print(f"  [{alert.severity.value.upper()}] {alert.item_id}: {alert.message}")
    
    return result


def demo_multi_scenario():
    """Demonstrate multiple scenario configurations."""
    print_header("MULTI-SCENARIO COMPARISON")
    
    # Create test inventory
    inventory = pd.DataFrame({
        "item_id": [f"SKU{i:03d}" for i in range(10)],
        "current_stock": [100, 50, 200, 30, 150, 80, 250, 10, 120, 60],
        "max_capacity": [500] * 10,
        "daily_demand_rate": [20, 15, 10, 25, 12, 18, 8, 30, 14, 22],
        "demand_std": [5, 4, 3, 7, 3, 5, 2, 10, 4, 6],
        "unit_cost": [50, 30, 20, 80, 25, 40, 15, 100, 35, 45],
        "revenue": [10000, 5000, 2000, 15000, 3000, 8000, 1000, 20000, 4000, 9000],
        "inventory_position": [100, 50, 200, 30, 150, 80, 250, 10, 120, 60],
    })
    
    scenarios = {
        "supplier_to_dc": {
            "policy_type": "periodic_review",
            "review_period": 7,
            "lead_time": 14,
            "service_level": 0.95,
        },
        "dc_to_store": {
            "policy_type": "periodic_review",
            "review_period": 1,
            "lead_time": 2,
            "service_level": 0.97,
        },
        "storage_to_picking": {
            "policy_type": "periodic_review",
            "review_period": 1,
            "lead_time": 0.5,
            "service_level": 0.99,
        },
    }
    
    print("Comparing replenishment across scenarios:")
    print("-" * 70)
    print(f"{'Scenario':<25} {'Review':>8} {'LT':>6} {'SL':>6} {'Orders':>8} {'Qty':>10}")
    print("-" * 70)
    
    for scenario_name, scenario_config in scenarios.items():
        config = {"scenarios": {scenario_name: scenario_config}}
        engine = ReplenishmentEngine(config_dict=config)
        
        result = engine.run(
            scenario=scenario_name,
            inventory_data=inventory.copy(),
        )
        
        print(
            f"{scenario_name:<25} "
            f"{scenario_config['review_period']:>8} "
            f"{scenario_config['lead_time']:>6} "
            f"{scenario_config['service_level']:>5.0%} "
            f"{result.summary['items_needing_order']:>8} "
            f"{result.summary['total_recommended_quantity']:>10,.0f}"
        )
    
    print("-" * 70)
    print("\nKey Insight: Different scenarios require different policies!")
    print("  - Longer lead times → More safety stock → Larger orders")
    print("  - Higher service levels → More safety stock")
    print("  - Shorter review periods → More frequent, smaller orders")


def main():
    """Run all demos."""
    print_header("UNIVERSAL REPLENISHMENT ENGINE", char="*")
    print("A comprehensive retail inventory replenishment solution")
    print("Supports: Supplier→DC, DC→Store, Store→DC, Storage→Pick, and more!")
    print()
    
    demos = [
        ("Classification Demo", demo_classification),
        ("Safety Stock Demo", demo_safety_stock),
        ("Policy Demo", demo_policies),
        ("Alert Demo", demo_alerts),
        ("Full Engine Demo", demo_full_engine),
        ("Multi-Scenario Demo", demo_multi_scenario),
    ]
    
    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. Run All")
    print(f"  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-7): ").strip()
            
            if choice == "0":
                print("\nThank you for exploring the Universal Replenishment Engine!")
                break
            
            choice = int(choice)
            
            if choice == len(demos) + 1:
                # Run all
                for name, func in demos:
                    func()
                print_header("DEMO COMPLETE", char="*")
            elif 1 <= choice <= len(demos):
                name, func = demos[choice - 1]
                func()
            else:
                print("Invalid choice. Please try again.")
        
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in demo: {e}")
            raise


if __name__ == "__main__":
    main()
