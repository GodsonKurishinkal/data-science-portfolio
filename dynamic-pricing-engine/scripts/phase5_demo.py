"""
Phase 5 Demonstration: Price Optimization Engine

This script demonstrates the complete price optimization workflow:
1. Loading elasticity data from Phase 3
2. Optimizing individual products
3. Portfolio-level optimization
4. Scenario comparison
5. Sensitivity analysis
6. Business insights and recommendations

Author: Godson Kurishinkal
Date: November 11, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.pricing.optimizer import PriceOptimizer, create_standard_scenarios
from src.pricing.elasticity import ElasticityAnalyzer


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format value as percentage."""
    return f"{value:+.2f}%"


def demo_single_product_optimization():
    """Demonstrate single product optimization."""
    print_section("1. SINGLE PRODUCT OPTIMIZATION")

    # Create sample product data
    product_data = {
        'product_id': 'LAPTOP_PRO_15',
        'current_price': 1200.0,
        'baseline_demand': 500.0,
        'elasticity': -1.8,
        'cost_per_unit': 750.0
    }

    print("Product Information:")
    print(f"  Product ID: {product_data['product_id']}")
    print(f"  Current Price: {format_currency(product_data['current_price'])}")
    print(f"  Baseline Demand: {product_data['baseline_demand']:.0f} units/month")
    print(f"  Price Elasticity: {product_data['elasticity']:.2f} (Elastic - price sensitive)")
    print(f"  Cost per Unit: {format_currency(product_data['cost_per_unit'])}")

    # Revenue optimization
    print_subsection("A. Revenue Optimization")

    revenue_optimizer = PriceOptimizer(objective='revenue', method='scipy')
    revenue_result = revenue_optimizer.optimize_single_product(
        product_id=product_data['product_id'],
        current_price=product_data['current_price'],
        baseline_demand=product_data['baseline_demand'],
        elasticity=product_data['elasticity'],
        constraints={'min_price': 900.0, 'max_price': 1500.0}
    )

    print("Optimization Results:")
    print(f"  Current Price: {format_currency(revenue_result['current_price'])}")
    print(f"  Optimal Price: {format_currency(revenue_result['optimal_price'])} ({format_percent(revenue_result['price_change_pct'])})")
    print(f"  Current Demand: {revenue_result['current_demand']:.0f} units")
    print(f"  Optimal Demand: {revenue_result['optimal_demand']:.0f} units ({format_percent(revenue_result['demand_change_pct'])})")
    print(f"  Current Revenue: {format_currency(revenue_result['current_revenue'])}")
    print(f"  Optimal Revenue: {format_currency(revenue_result['optimal_revenue'])} ({format_percent(revenue_result['revenue_change_pct'])})")
    print(f"  Revenue Gain: {format_currency(revenue_result['optimal_revenue'] - revenue_result['current_revenue'])}")

    # Profit optimization
    print_subsection("B. Profit Optimization")

    profit_optimizer = PriceOptimizer(objective='profit', method='scipy')
    profit_result = profit_optimizer.optimize_single_product(
        product_id=product_data['product_id'],
        current_price=product_data['current_price'],
        baseline_demand=product_data['baseline_demand'],
        elasticity=product_data['elasticity'],
        cost_per_unit=product_data['cost_per_unit'],
        constraints={'min_price': 900.0, 'max_price': 1500.0, 'min_margin_pct': 25.0}
    )

    print("Optimization Results:")
    print(f"  Current Price: {format_currency(profit_result['current_price'])}")
    print(f"  Optimal Price: {format_currency(profit_result['optimal_price'])} ({format_percent(profit_result['price_change_pct'])})")
    print(f"  Current Profit: {format_currency(profit_result['current_profit'])}")
    print(f"  Optimal Profit: {format_currency(profit_result['optimal_profit'])} ({format_percent(profit_result['profit_change_pct'])})")
    print(f"  Profit Gain: {format_currency(profit_result['optimal_profit'] - profit_result['current_profit'])}")
    print(f"  Current Margin: {profit_result['current_margin_pct']:.1f}%")
    print(f"  Optimal Margin: {profit_result['optimal_margin_pct']:.1f}%")

    # Compare objectives
    print_subsection("C. Revenue vs Profit Optimization Comparison")

    comparison_df = pd.DataFrame({
        'Metric': ['Optimal Price', 'Price Change', 'Demand', 'Revenue', 'Profit', 'Margin %'],
        'Revenue Optimization': [
            format_currency(revenue_result['optimal_price']),
            format_percent(revenue_result['price_change_pct']),
            f"{revenue_result['optimal_demand']:.0f}",
            format_currency(revenue_result['optimal_revenue']),
            format_currency(profit_result['optimal_profit']) if 'optimal_profit' in profit_result else 'N/A',
            f"{profit_result['optimal_margin_pct']:.1f}%" if 'optimal_margin_pct' in profit_result else 'N/A'
        ],
        'Profit Optimization': [
            format_currency(profit_result['optimal_price']),
            format_percent(profit_result['price_change_pct']),
            f"{profit_result['optimal_demand']:.0f}",
            format_currency(profit_result['optimal_revenue']),
            format_currency(profit_result['optimal_profit']),
            f"{profit_result['optimal_margin_pct']:.1f}%"
        ]
    })

    print(comparison_df.to_string(index=False))

    print("\nKey Insight: Profit optimization typically results in higher prices and margins,")
    print("while revenue optimization may sacrifice margin for volume.")


def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    print_section("2. PORTFOLIO OPTIMIZATION")

    # Create diverse product portfolio
    portfolio_df = pd.DataFrame({
        'product_id': [
            'LAPTOP_PRO_15', 'LAPTOP_AIR_13', 'TABLET_10',
            'PHONE_X', 'HEADPHONES_PRO', 'MOUSE_WIRELESS'
        ],
        'current_price': [1200.0, 999.0, 499.0, 899.0, 349.0, 79.0],
        'baseline_demand': [500.0, 800.0, 1500.0, 2000.0, 3000.0, 5000.0],
        'elasticity': [-1.8, -1.5, -1.2, -2.0, -0.8, -0.5],
        'cost_per_unit': [750.0, 600.0, 300.0, 550.0, 180.0, 35.0]
    })

    print(f"Portfolio: {len(portfolio_df)} products")
    print("\nCurrent Portfolio Status:")
    print(portfolio_df[['product_id', 'current_price', 'baseline_demand', 'elasticity']].to_string(index=False))

    # Optimize portfolio for profit
    print_subsection("A. Profit Optimization with Constraints")

    optimizer = PriceOptimizer(objective='profit', method='scipy')

    # Apply business constraints
    constraints = {
        'min_margin_pct': 20.0,  # Minimum 20% margin
        'max_discount_pct': 15.0  # Maximum 15% discount
    }

    results_df = optimizer.optimize_portfolio(portfolio_df, constraints=constraints)

    # Display results
    print("\nOptimization Results:")
    display_cols = ['product_id', 'current_price', 'optimal_price', 'price_change_pct',
                    'current_profit', 'optimal_profit', 'profit_change_pct']
    print(results_df[display_cols].to_string(index=False))

    # Portfolio-level metrics
    print_subsection("B. Portfolio-Level Impact")

    total_current_revenue = results_df['current_revenue'].sum()
    total_optimal_revenue = results_df['optimal_revenue'].sum()
    total_current_profit = results_df['current_profit'].sum()
    total_optimal_profit = results_df['optimal_profit'].sum()

    revenue_gain = total_optimal_revenue - total_current_revenue
    profit_gain = total_optimal_profit - total_current_profit

    print(f"Total Current Revenue: {format_currency(total_current_revenue)}")
    print(f"Total Optimal Revenue: {format_currency(total_optimal_revenue)}")
    print(f"Revenue Gain: {format_currency(revenue_gain)} ({format_percent((revenue_gain / total_current_revenue) * 100)})")
    print()
    print(f"Total Current Profit: {format_currency(total_current_profit)}")
    print(f"Total Optimal Profit: {format_currency(total_optimal_profit)}")
    print(f"Profit Gain: {format_currency(profit_gain)} ({format_percent((profit_gain / total_current_profit) * 100)})")

    # Elasticity-based insights
    print_subsection("C. Elasticity Segmentation & Recommendations")

    segmentation = optimizer.calculate_price_elasticity_segments(portfolio_df)

    print("Product Segments by Elasticity:")
    print(segmentation['summary'].to_string())

    print("\nRecommendations by Segment:")
    for segment, recommendation in segmentation['recommendations'].items():
        if segment in segmentation['segments']['elasticity_segment'].values:
            print(f"\n{segment}:")
            print(f"  {recommendation}")


def demo_scenario_simulation():
    """Demonstrate scenario simulation."""
    print_section("3. SCENARIO SIMULATION")

    # Select a product for detailed analysis
    product = {
        'product_id': 'HEADPHONES_PRO',
        'current_price': 349.0,
        'baseline_demand': 3000.0,
        'elasticity': -0.8,
        'cost_per_unit': 180.0
    }

    print(f"Analyzing: {product['product_id']}")
    print(f"Current Price: {format_currency(product['current_price'])}")
    print(f"Elasticity: {product['elasticity']:.2f} (Inelastic - less price sensitive)")

    # Simulate standard pricing scenarios
    print_subsection("A. Standard Pricing Scenarios")

    optimizer = PriceOptimizer(objective='profit', method='scipy')
    scenarios_df = optimizer.simulate_scenarios(
        product_id=product['product_id'],
        current_price=product['current_price'],
        baseline_demand=product['baseline_demand'],
        elasticity=product['elasticity'],
        cost_per_unit=product['cost_per_unit']
    )

    # Display all scenarios
    display_cols = ['scenario', 'price', 'predicted_demand', 'revenue', 'profit', 'margin_pct']
    print(scenarios_df[display_cols].to_string(index=False))

    # Find best scenario
    best_profit_idx = scenarios_df['profit'].idxmax()
    best_scenario = scenarios_df.loc[best_profit_idx]

    print(f"\nBest Scenario: {best_scenario['scenario']}")
    print(f"  Price: {format_currency(best_scenario['price'])}")
    print(f"  Profit: {format_currency(best_scenario['profit'])}")
    print(f"  Margin: {best_scenario['margin_pct']:.1f}%")

    # Custom competitive scenarios
    print_subsection("B. Competitive Pricing Scenarios")

    competitive_scenarios = [
        {'name': 'Match Competitor A', 'price': 329.0},
        {'name': 'Undercut Competitor B', 'price': 299.0},
        {'name': 'Premium Positioning', 'price': 399.0},
        {'name': 'Psychological ($9.99)', 'price': 349.99},
        {'name': 'Bundle Discount', 'price': 314.10}  # 10% off for bundle
    ]

    competitive_df = optimizer.simulate_scenarios(
        product_id=product['product_id'],
        current_price=product['current_price'],
        baseline_demand=product['baseline_demand'],
        elasticity=product['elasticity'],
        cost_per_unit=product['cost_per_unit'],
        scenarios=competitive_scenarios
    )

    print(competitive_df[display_cols].to_string(index=False))


def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis."""
    print_section("4. SENSITIVITY ANALYSIS")

    # Product for sensitivity analysis
    product = {
        'product_id': 'TABLET_10',
        'current_price': 499.0,
        'baseline_demand': 1500.0,
        'elasticity': -1.2,
        'cost_per_unit': 300.0
    }

    print(f"Product: {product['product_id']}")
    print(f"Current Price: {format_currency(product['current_price'])}")
    print(f"Elasticity: {product['elasticity']:.2f} (Moderately elastic)")

    # Generate sensitivity curves
    print_subsection("A. Price-Response Curves")

    optimizer = PriceOptimizer(objective='profit', method='scipy')
    sensitivity_df = optimizer.sensitivity_analysis(
        product_id=product['product_id'],
        current_price=product['current_price'],
        baseline_demand=product['baseline_demand'],
        elasticity=product['elasticity'],
        price_range=(350.0, 650.0),
        n_points=30,
        cost_per_unit=product['cost_per_unit']
    )

    # Find optimal points
    max_revenue_idx = sensitivity_df['revenue'].idxmax()
    max_profit_idx = sensitivity_df['profit'].idxmax()

    optimal_revenue_price = sensitivity_df.loc[max_revenue_idx, 'price']
    optimal_revenue = sensitivity_df.loc[max_revenue_idx, 'revenue']
    optimal_profit_price = sensitivity_df.loc[max_profit_idx, 'price']
    optimal_profit = sensitivity_df.loc[max_profit_idx, 'profit']

    print(f"\nOptimal Revenue Price: {format_currency(optimal_revenue_price)}")
    print(f"  Maximum Revenue: {format_currency(optimal_revenue)}")

    print(f"\nOptimal Profit Price: {format_currency(optimal_profit_price)}")
    print(f"  Maximum Profit: {format_currency(optimal_profit)}")

    # Show sample points from curve
    print_subsection("B. Sample Points from Sensitivity Curve")

    # Select every 5th point for display
    sample_df = sensitivity_df.iloc[::5].copy()
    display_cols = ['price', 'price_change_pct', 'predicted_demand', 'revenue', 'profit', 'margin_pct']
    print(sample_df[display_cols].to_string(index=False))

    # Key insights
    print_subsection("C. Key Insights")

    current_price = product['current_price']
    current_idx = (sensitivity_df['price'] - current_price).abs().idxmin()
    current_profit = sensitivity_df.loc[current_idx, 'profit']

    profit_improvement = optimal_profit - current_profit
    profit_improvement_pct = (profit_improvement / current_profit) * 100

    print(f"Current Price: {format_currency(current_price)}")
    print(f"  Current Profit: {format_currency(current_profit)}")

    print(f"\nOptimal Price: {format_currency(optimal_profit_price)}")
    print(f"  Optimal Profit: {format_currency(optimal_profit)}")
    print(f"  Improvement: {format_currency(profit_improvement)} ({format_percent(profit_improvement_pct)})")

    price_change = optimal_profit_price - current_price
    price_change_pct = (price_change / current_price) * 100

    print(f"\nPrice Adjustment: {format_currency(price_change)} ({format_percent(price_change_pct)})")

    if price_change_pct < 0:
        print("Recommendation: DECREASE price to maximize profit through higher volume")
    else:
        print("Recommendation: INCREASE price to maximize profit through higher margins")


def demo_optimization_summary():
    """Demonstrate optimization summary and aggregated metrics."""
    print_section("5. OPTIMIZATION SUMMARY & BUSINESS INSIGHTS")

    # Create optimizer and run multiple optimizations
    optimizer = PriceOptimizer(objective='profit', method='scipy')

    # Portfolio of products
    products = [
        {'product_id': 'PROD_001', 'current_price': 50.0, 'baseline_demand': 1000.0, 'elasticity': -1.5, 'cost_per_unit': 20.0},
        {'product_id': 'PROD_002', 'current_price': 30.0, 'baseline_demand': 2000.0, 'elasticity': -0.8, 'cost_per_unit': 15.0},
        {'product_id': 'PROD_003', 'current_price': 100.0, 'baseline_demand': 500.0, 'elasticity': -2.0, 'cost_per_unit': 40.0},
        {'product_id': 'PROD_004', 'current_price': 75.0, 'baseline_demand': 800.0, 'elasticity': -1.2, 'cost_per_unit': 35.0},
        {'product_id': 'PROD_005', 'current_price': 25.0, 'baseline_demand': 3000.0, 'elasticity': -0.5, 'cost_per_unit': 12.0},
    ]

    print("Optimizing 5 products...")

    for product in products:
        optimizer.optimize_single_product(**product)

    # Get summary
    summary = optimizer.get_optimization_summary()

    print_subsection("A. Aggregate Optimization Metrics")

    print(f"Total Optimizations: {summary['total_optimizations']}")
    print(f"Products Optimized: {summary['products_optimized']}")
    print(f"Method Used: {summary['method_used'].upper()}")
    print(f"Objective: {summary['objective_used'].upper()}")

    print(f"\nAverage Price Change: {format_percent(summary['avg_price_change_pct'])}")
    print(f"Average Revenue Change: {format_percent(summary['avg_revenue_change_pct'])}")
    print(f"Average Profit Change: {format_percent(summary['avg_profit_change_pct'])}")

    print(f"\nTotal Current Revenue: {format_currency(summary['total_current_revenue'])}")
    print(f"Total Optimal Revenue: {format_currency(summary['total_optimal_revenue'])}")
    print(f"Total Revenue Gain: {format_currency(summary['total_revenue_gain'])}")

    print(f"\nTotal Current Profit: {format_currency(summary['total_current_profit'])}")
    print(f"Total Optimal Profit: {format_currency(summary['total_optimal_profit'])}")
    print(f"Total Profit Gain: {format_currency(summary['total_profit_gain'])}")

    print_subsection("B. Business Impact")

    monthly_profit_gain = summary['total_profit_gain']
    annual_profit_gain = monthly_profit_gain * 12

    print(f"Monthly Profit Increase: {format_currency(monthly_profit_gain)}")
    print(f"Annual Profit Increase: {format_currency(annual_profit_gain)}")

    print_subsection("C. Strategic Recommendations")

    if summary['avg_price_change_pct'] < -5:
        print("• Portfolio shows significant price reduction opportunity")
        print("• Products are generally elastic - volume strategy recommended")
        print("• Consider aggressive promotions and discounts")
    elif summary['avg_price_change_pct'] > 5:
        print("• Portfolio shows pricing power - price increases recommended")
        print("• Products are generally inelastic - margin strategy recommended")
        print("• Focus on premium positioning and value communication")
    else:
        print("• Portfolio is near optimal pricing")
        print("• Focus on cost reduction and operational efficiency")
        print("• Consider differentiation strategies")

    print(f"\n• Average profit improvement: {summary['avg_profit_change_pct']:.1f}%")
    print("• Regular price optimization recommended (monthly or quarterly)")
    print("• Monitor elasticity changes due to competition and market conditions")


def main():
    """Run all Phase 5 demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 20 + "PHASE 5: PRICE OPTIMIZATION ENGINE")
    print(" " * 25 + "Demonstration Script")
    print("=" * 80)

    print("\nThis demonstration showcases advanced price optimization capabilities:")
    print("  • Revenue and profit maximization")
    print("  • Multiple optimization algorithms (scipy, grid search, gradient descent)")
    print("  • Business constraint handling (price bounds, margins, discounts)")
    print("  • Portfolio-level optimization")
    print("  • Scenario simulation and comparison")
    print("  • Sensitivity analysis with response curves")
    print("  • Elasticity-based segmentation")
    print("  • Aggregated business insights")

    try:
        # Run demonstrations
        demo_single_product_optimization()
        demo_portfolio_optimization()
        demo_scenario_simulation()
        demo_sensitivity_analysis()
        demo_optimization_summary()

        # Conclusion
        print_section("DEMONSTRATION COMPLETE")
        print("Phase 5 Price Optimization Engine successfully demonstrates:")
        print("  ✓ Advanced mathematical optimization techniques")
        print("  ✓ Real-world business constraint handling")
        print("  ✓ Multiple objective functions (revenue vs profit)")
        print("  ✓ Portfolio-scale optimization capabilities")
        print("  ✓ Scenario analysis and sensitivity testing")
        print("  ✓ Data-driven pricing recommendations")
        print("  ✓ Clear business value quantification")

        print("\nThis system enables automated, data-driven pricing decisions that can")
        print("significantly improve profitability while maintaining competitive positioning.")

        print("\n" + "=" * 80)
        print(" " * 30 + "END OF DEMO")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
