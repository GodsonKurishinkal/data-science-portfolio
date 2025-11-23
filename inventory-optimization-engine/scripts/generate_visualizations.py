#!/usr/bin/env python3
"""Generate visualizations for Inventory Optimization Engine."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DemandCalculator
from src.inventory import ABCAnalyzer, SafetyStockCalculator, EOQCalculator
from src.optimization import CostCalculator
from src.utils import load_config, setup_logging, ensure_directory


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("Set2")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_abc_xyz_matrix(classified_data, output_path):
    """
    Create ABC-XYZ classification matrix heatmap.
    
    Args:
        classified_data: DataFrame with ABC-XYZ classification
        output_path: Path to save figure
    """
    print("📊 Generating ABC-XYZ Matrix...")
    
    # Create pivot table for heatmap
    matrix = classified_data.groupby(['abc_class', 'xyz_class']).size().unstack(fill_value=0)
    
    # Calculate percentages
    matrix_pct = (matrix / matrix.sum().sum() * 100).round(1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count heatmap
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Number of Items'})
    ax1.set_title('ABC-XYZ Classification Matrix\n(Item Count)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('XYZ Class (Variability)', fontsize=12)
    ax1.set_ylabel('ABC Class (Value)', fontsize=12)
    
    # Percentage heatmap
    sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('ABC-XYZ Classification Matrix\n(% of Total Items)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('XYZ Class (Variability)', fontsize=12)
    ax2.set_ylabel('ABC Class (Value)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'abc_xyz_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'abc_xyz_matrix.png'}")
    plt.close()


def plot_revenue_distribution(classified_data, output_path):
    """
    Plot revenue distribution by ABC-XYZ class.
    
    Args:
        classified_data: DataFrame with ABC-XYZ classification
        output_path: Path to save figure
    """
    print("📊 Generating Revenue Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Revenue by ABC class (Pareto chart)
    abc_revenue = classified_data.groupby('abc_class')['revenue_sum'].sum().sort_values(ascending=False)
    abc_revenue_pct = (abc_revenue / abc_revenue.sum() * 100)
    cumulative_pct = abc_revenue_pct.cumsum()
    
    ax1 = axes[0, 0]
    x = np.arange(len(abc_revenue))
    bars = ax1.bar(x, abc_revenue / 1e6, color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_xlabel('ABC Class', fontsize=12)
    ax1.set_ylabel('Revenue ($ Millions)', fontsize=12, color='tab:blue')
    ax1.set_title('Revenue by ABC Class (Pareto)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(abc_revenue.index)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add percentage values on bars
    for i, (bar, pct) in enumerate(zip(bars, abc_revenue_pct)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Cumulative line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, cumulative_pct, 'ro-', linewidth=2, markersize=8, label='Cumulative %')
    ax1_twin.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='tab:red')
    ax1_twin.tick_params(axis='y', labelcolor='tab:red')
    ax1_twin.set_ylim([0, 105])
    ax1_twin.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax1_twin.legend(loc='upper left')
    
    # 2. Revenue by combined ABC-XYZ class
    ax2 = axes[0, 1]
    abcxyz_revenue = classified_data.groupby('abc_xyz_class')['revenue_sum'].sum().sort_values(ascending=False).head(15)
    colors = ['#d62728' if x.startswith('A') else '#ff7f0e' if x.startswith('B') else '#2ca02c' 
              for x in abcxyz_revenue.index]
    ax2.barh(range(len(abcxyz_revenue)), abcxyz_revenue / 1e6, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(abcxyz_revenue)))
    ax2.set_yticklabels(abcxyz_revenue.index)
    ax2.set_xlabel('Revenue ($ Millions)', fontsize=12)
    ax2.set_title('Top 15 ABC-XYZ Classes by Revenue', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Item count by class
    ax3 = axes[1, 0]
    class_counts = classified_data['abc_xyz_class'].value_counts().sort_values(ascending=False).head(15)
    colors = ['#d62728' if x.startswith('A') else '#ff7f0e' if x.startswith('B') else '#2ca02c' 
              for x in class_counts.index]
    ax3.barh(range(len(class_counts)), class_counts, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(class_counts)))
    ax3.set_yticklabels(class_counts.index)
    ax3.set_xlabel('Number of Items', fontsize=12)
    ax3.set_title('Top 15 ABC-XYZ Classes by Item Count', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Scatter: Revenue vs CV
    ax4 = axes[1, 1]
    for abc_class in ['A', 'B', 'C']:
        subset = classified_data[classified_data['abc_class'] == abc_class]
        ax4.scatter(subset['demand_cv'], subset['revenue_sum'] / 1000, 
                   alpha=0.5, s=30, label=f'Class {abc_class}')
    ax4.set_xlabel('Coefficient of Variation (Demand Variability)', fontsize=12)
    ax4.set_ylabel('Revenue ($ Thousands)', fontsize=12)
    ax4.set_title('Revenue vs Demand Variability', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, min(5, classified_data['demand_cv'].quantile(0.95))])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'revenue_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'revenue_distribution.png'}")
    plt.close()


def plot_inventory_metrics(inventory_data, output_path):
    """
    Plot inventory optimization metrics.
    
    Args:
        inventory_data: DataFrame with inventory calculations
        output_path: Path to save figure
    """
    print("📊 Generating Inventory Metrics...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. EOQ Distribution by ABC class
    ax1 = fig.add_subplot(gs[0, 0])
    inventory_data.boxplot(column='eoq', by='abc_class', ax=ax1)
    ax1.set_title('EOQ Distribution by ABC Class', fontsize=12, fontweight='bold')
    ax1.set_xlabel('ABC Class', fontsize=10)
    ax1.set_ylabel('Economic Order Quantity', fontsize=10)
    plt.suptitle('')  # Remove default title
    
    # 2. Safety Stock Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    inventory_data.boxplot(column='safety_stock', by='abc_class', ax=ax2)
    ax2.set_title('Safety Stock by ABC Class', fontsize=12, fontweight='bold')
    ax2.set_xlabel('ABC Class', fontsize=10)
    ax2.set_ylabel('Safety Stock (units)', fontsize=10)
    plt.suptitle('')
    
    # 3. Reorder Point Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    inventory_data.boxplot(column='reorder_point', by='abc_class', ax=ax3)
    ax3.set_title('Reorder Point by ABC Class', fontsize=12, fontweight='bold')
    ax3.set_xlabel('ABC Class', fontsize=10)
    ax3.set_ylabel('Reorder Point (units)', fontsize=10)
    plt.suptitle('')
    
    # 4. Average inventory levels by class
    ax4 = fig.add_subplot(gs[1, 0])
    avg_inventory = inventory_data.groupby('abc_class')['average_inventory'].mean()
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax4.bar(avg_inventory.index, avg_inventory, color=colors, alpha=0.7)
    ax4.set_title('Average Inventory Level by ABC Class', fontsize=12, fontweight='bold')
    ax4.set_xlabel('ABC Class', fontsize=10)
    ax4.set_ylabel('Average Inventory (units)', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Order frequency by class
    ax5 = fig.add_subplot(gs[1, 1])
    avg_orders = inventory_data.groupby('abc_class')['orders_per_year'].mean()
    bars = ax5.bar(avg_orders.index, avg_orders, color=colors, alpha=0.7)
    ax5.set_title('Average Orders per Year by ABC Class', fontsize=12, fontweight='bold')
    ax5.set_xlabel('ABC Class', fontsize=10)
    ax5.set_ylabel('Orders per Year', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Days between orders
    ax6 = fig.add_subplot(gs[1, 2])
    avg_days = inventory_data.groupby('abc_class')['days_between_orders'].mean()
    bars = ax6.bar(avg_days.index, avg_days, color=colors, alpha=0.7)
    ax6.set_title('Average Days Between Orders', fontsize=12, fontweight='bold')
    ax6.set_xlabel('ABC Class', fontsize=10)
    ax6.set_ylabel('Days', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Scatter: EOQ vs Demand
    ax7 = fig.add_subplot(gs[2, :2])
    for abc_class in ['A', 'B', 'C']:
        subset = inventory_data[inventory_data['abc_class'] == abc_class]
        ax7.scatter(subset['sales_mean'], subset['eoq'], 
                   alpha=0.5, s=30, label=f'Class {abc_class}')
    ax7.set_xlabel('Average Daily Demand', fontsize=10)
    ax7.set_ylabel('Economic Order Quantity', fontsize=10)
    ax7.set_title('EOQ vs Average Demand by ABC Class', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_stats = inventory_data.groupby('abc_class').agg({
        'eoq': 'mean',
        'safety_stock': 'mean',
        'reorder_point': 'mean',
        'orders_per_year': 'mean'
    }).round(0)
    
    table_data = []
    table_data.append(['Metric', 'A', 'B', 'C'])
    table_data.append(['EOQ', f"{summary_stats.loc['A', 'eoq']:.0f}", 
                      f"{summary_stats.loc['B', 'eoq']:.0f}", 
                      f"{summary_stats.loc['C', 'eoq']:.0f}"])
    table_data.append(['Safety Stock', f"{summary_stats.loc['A', 'safety_stock']:.0f}", 
                      f"{summary_stats.loc['B', 'safety_stock']:.0f}", 
                      f"{summary_stats.loc['C', 'safety_stock']:.0f}"])
    table_data.append(['Reorder Point', f"{summary_stats.loc['A', 'reorder_point']:.0f}", 
                      f"{summary_stats.loc['B', 'reorder_point']:.0f}", 
                      f"{summary_stats.loc['C', 'reorder_point']:.0f}"])
    table_data.append(['Orders/Year', f"{summary_stats.loc['A', 'orders_per_year']:.1f}", 
                      f"{summary_stats.loc['B', 'orders_per_year']:.1f}", 
                      f"{summary_stats.loc['C', 'orders_per_year']:.1f}"])
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Summary Statistics by ABC Class', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path / 'inventory_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'inventory_metrics.png'}")
    plt.close()


def plot_cost_analysis(inventory_data, output_path):
    """
    Plot cost analysis and breakdown.
    
    Args:
        inventory_data: DataFrame with cost calculations
        output_path: Path to save figure
    """
    print("📊 Generating Cost Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total cost breakdown by ABC class
    ax1 = axes[0, 0]
    cost_by_class = inventory_data.groupby('abc_class').agg({
        'annual_holding_cost': 'sum',
        'annual_ordering_cost': 'sum',
        'annual_purchase_cost': 'sum'
    })
    
    x = np.arange(len(cost_by_class))
    width = 0.25
    
    ax1.bar(x - width, cost_by_class['annual_holding_cost'] / 1e6, width, 
           label='Holding Cost', color='#ff7f0e', alpha=0.8)
    ax1.bar(x, cost_by_class['annual_ordering_cost'] / 1e6, width, 
           label='Ordering Cost', color='#2ca02c', alpha=0.8)
    ax1.bar(x + width, cost_by_class['annual_purchase_cost'] / 1e6, width, 
           label='Purchase Cost', color='#d62728', alpha=0.8)
    
    ax1.set_xlabel('ABC Class', fontsize=12)
    ax1.set_ylabel('Annual Cost ($ Millions)', fontsize=12)
    ax1.set_title('Cost Breakdown by ABC Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cost_by_class.index)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cost composition pie chart
    ax2 = axes[0, 1]
    total_costs = inventory_data[['annual_holding_cost', 'annual_ordering_cost', 
                                   'annual_purchase_cost']].sum()
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    explode = (0.05, 0.05, 0)
    
    wedges, texts, autotexts = ax2.pie(total_costs, labels=['Holding', 'Ordering', 'Purchase'],
                                        autopct='%1.1f%%', colors=colors, explode=explode,
                                        startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Overall Cost Composition', fontsize=14, fontweight='bold')
    
    # 3. Total annual cost by class
    ax3 = axes[1, 0]
    total_cost_by_class = inventory_data.groupby('abc_class')['total_annual_cost'].sum() / 1e6
    colors_abc = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax3.bar(total_cost_by_class.index, total_cost_by_class, color=colors_abc, alpha=0.7)
    ax3.set_xlabel('ABC Class', fontsize=12)
    ax3.set_ylabel('Total Annual Cost ($ Millions)', fontsize=12)
    ax3.set_title('Total Annual Cost by ABC Class', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total = total_cost_by_class.sum()
    for bar, value in zip(bars, total_cost_by_class):
        height = bar.get_height()
        pct = (value / total * 100)
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${value:.1f}M\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 4. Cost per unit by class
    ax4 = axes[1, 1]
    inventory_data['cost_per_unit'] = (
        inventory_data['annual_holding_cost'] + inventory_data['annual_ordering_cost']
    ) / inventory_data['annual_demand']
    
    cost_per_unit = inventory_data.groupby('abc_class')['cost_per_unit'].mean()
    bars = ax4.bar(cost_per_unit.index, cost_per_unit, color=colors_abc, alpha=0.7)
    ax4.set_xlabel('ABC Class', fontsize=12)
    ax4.set_ylabel('Operational Cost per Unit ($)', fontsize=12)
    ax4.set_title('Average Operational Cost per Unit', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'cost_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'cost_analysis.png'}")
    plt.close()


def plot_service_level_tradeoff(inventory_data, output_path):
    """
    Plot service level vs cost tradeoff.
    
    Args:
        inventory_data: DataFrame with inventory data
        output_path: Path to save figure
    """
    print("📊 Generating Service Level Tradeoff...")
    
    from scipy import stats
    
    # Calculate costs for different service levels
    service_levels = np.arange(0.80, 0.995, 0.01)
    
    # Use sample item statistics
    sample_stats = inventory_data.iloc[0]
    demand_std = sample_stats['sales_std']
    unit_cost = sample_stats['sell_price_mean']
    holding_rate = 0.25
    
    results = []
    for sl in service_levels:
        z = stats.norm.ppf(sl)
        safety_stock = z * demand_std * np.sqrt(7)  # 7-day lead time
        holding_cost = safety_stock * unit_cost * holding_rate
        
        # Simplified stockout cost
        stockout_prob = 1 - sl
        stockout_cost = stockout_prob * unit_cost * 2  # 2x penalty
        
        total_cost = holding_cost + stockout_cost
        results.append({
            'service_level': sl,
            'safety_stock': safety_stock,
            'holding_cost': holding_cost,
            'stockout_cost': stockout_cost,
            'total_cost': total_cost
        })
    
    df_tradeoff = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Cost vs Service Level
    ax1.plot(df_tradeoff['service_level'] * 100, df_tradeoff['holding_cost'], 
            'b-', linewidth=2, label='Holding Cost')
    ax1.plot(df_tradeoff['service_level'] * 100, df_tradeoff['stockout_cost'], 
            'r-', linewidth=2, label='Stockout Cost')
    ax1.plot(df_tradeoff['service_level'] * 100, df_tradeoff['total_cost'], 
            'g-', linewidth=3, label='Total Cost', alpha=0.7)
    
    # Mark optimal point
    optimal_idx = df_tradeoff['total_cost'].idxmin()
    optimal_sl = df_tradeoff.loc[optimal_idx, 'service_level'] * 100
    optimal_cost = df_tradeoff.loc[optimal_idx, 'total_cost']
    ax1.plot(optimal_sl, optimal_cost, 'ko', markersize=10, label=f'Optimal ({optimal_sl:.1f}%)')
    
    ax1.set_xlabel('Service Level (%)', fontsize=12)
    ax1.set_ylabel('Annual Cost ($)', fontsize=12)
    ax1.set_title('Service Level vs Cost Tradeoff', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=95, color='gray', linestyle='--', alpha=0.5, label='95% SL')
    
    # 2. Safety Stock vs Service Level
    ax2.plot(df_tradeoff['service_level'] * 100, df_tradeoff['safety_stock'], 
            'purple', linewidth=2)
    ax2.fill_between(df_tradeoff['service_level'] * 100, df_tradeoff['safety_stock'], 
                     alpha=0.3, color='purple')
    ax2.set_xlabel('Service Level (%)', fontsize=12)
    ax2.set_ylabel('Safety Stock (units)', fontsize=12)
    ax2.set_title('Required Safety Stock by Service Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark common service levels
    for sl in [0.90, 0.95, 0.99]:
        ss = df_tradeoff[df_tradeoff['service_level'].between(sl-0.001, sl+0.001)]['safety_stock'].iloc[0]
        ax2.plot(sl * 100, ss, 'ro', markersize=8)
        ax2.annotate(f'{sl*100:.0f}%\n{ss:.0f} units', 
                    xy=(sl * 100, ss), xytext=(sl * 100, ss + 2),
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'service_level_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'service_level_tradeoff.png'}")
    plt.close()


def plot_top_items_dashboard(inventory_data, output_path):
    """
    Create dashboard for top items.
    
    Args:
        inventory_data: DataFrame with inventory data
        output_path: Path to save figure
    """
    print("📊 Generating Top Items Dashboard...")
    
    # Get top 20 items by revenue
    top_items = inventory_data.nlargest(20, 'revenue_sum').copy()
    top_items['item_store'] = top_items['item_id'] + '\n' + top_items['store_id']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Revenue ranking
    ax1 = axes[0, 0]
    colors = ['#d62728' if x == 'A' else '#ff7f0e' if x == 'B' else '#2ca02c' 
              for x in top_items['abc_class']]
    bars = ax1.barh(range(len(top_items)), top_items['revenue_sum'] / 1000, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_items)))
    ax1.set_yticklabels(top_items['item_store'], fontsize=8)
    ax1.set_xlabel('Revenue ($ Thousands)', fontsize=12)
    ax1.set_title('Top 20 Items by Revenue', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add ABC class labels
    for i, (bar, abc) in enumerate(zip(bars, top_items['abc_class'])):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, abc, 
                ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 2. EOQ comparison
    ax2 = axes[0, 1]
    bars = ax2.barh(range(len(top_items)), top_items['eoq'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_items)))
    ax2.set_yticklabels(top_items['item_store'], fontsize=8)
    ax2.set_xlabel('Economic Order Quantity', fontsize=12)
    ax2.set_title('EOQ for Top 20 Items', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Safety Stock requirements
    ax3 = axes[1, 0]
    bars = ax3.barh(range(len(top_items)), top_items['safety_stock'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_items)))
    ax3.set_yticklabels(top_items['item_store'], fontsize=8)
    ax3.set_xlabel('Safety Stock (units)', fontsize=12)
    ax3.set_title('Safety Stock for Top 20 Items', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Annual cost
    ax4 = axes[1, 1]
    bars = ax4.barh(range(len(top_items)), top_items['total_annual_cost'] / 1000, 
                   color=colors, alpha=0.7)
    ax4.set_yticks(range(len(top_items)))
    ax4.set_yticklabels(top_items['item_store'], fontsize=8)
    ax4.set_xlabel('Total Annual Cost ($ Thousands)', fontsize=12)
    ax4.set_title('Total Annual Cost for Top 20 Items', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path / 'top_items_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'top_items_dashboard.png'}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("📊 GENERATING VISUALIZATIONS FOR INVENTORY OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Setup
    setup_logging(level='INFO')
    setup_plot_style()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config('config/config.yaml')
    
    # Create output directory
    output_path = Path(config['visualization']['output_path'])
    ensure_directory(output_path)
    print(f"✅ Output directory: {output_path}")
    print()
    
    # Load and process data
    print("📊 Loading M5 data...")
    loader = DataLoader(config['data']['raw_data_path'])
    raw_data = loader.load_raw_data()
    full_data = loader.process_data()
    
    # Use sample stores for visualization
    sample_stores = ['CA_1', 'TX_1']
    data = loader.filter_data(
        full_data,
        stores=sample_stores,
        start_date='2015-01-01',
        end_date='2016-03-27'
    )
    print(f"✅ Loaded {len(data):,} records")
    print()
    
    # Calculate demand statistics
    print("📈 Calculating demand statistics...")
    calc = DemandCalculator()
    demand_stats = calc.calculate_demand_statistics(data, group_cols=['store_id', 'item_id'])
    print()
    
    # Perform ABC/XYZ classification
    print("🎯 Performing ABC/XYZ classification...")
    abc_analyzer = ABCAnalyzer(
        abc_thresholds=config['inventory']['abc_thresholds'],
        xyz_thresholds=config['inventory']['xyz_thresholds']
    )
    classified = abc_analyzer.perform_combined_analysis(demand_stats)
    print()
    
    # Calculate inventory parameters
    print("💰 Calculating inventory parameters...")
    eoq_calc = EOQCalculator(
        ordering_cost=config['inventory']['costs']['ordering_cost'],
        holding_cost_rate=config['inventory']['costs']['holding_cost_rate']
    )
    inventory_data = eoq_calc.calculate_for_dataframe(classified)
    
    ss_calc = SafetyStockCalculator(
        service_level=0.95,
        lead_time=config['inventory']['lead_time']['default']
    )
    inventory_data = ss_calc.calculate_for_dataframe(inventory_data, method='basic')
    
    # Calculate reorder points
    from src.inventory import ReorderPointCalculator
    rop_calc = ReorderPointCalculator(lead_time=config['inventory']['lead_time']['default'])
    inventory_data = rop_calc.calculate_for_dataframe(inventory_data)
    print()
    
    # Generate all visualizations
    print("🎨 Generating visualizations...")
    print()
    
    plot_abc_xyz_matrix(classified, output_path)
    plot_revenue_distribution(classified, output_path)
    plot_inventory_metrics(inventory_data, output_path)
    plot_cost_analysis(inventory_data, output_path)
    plot_service_level_tradeoff(inventory_data, output_path)
    plot_top_items_dashboard(inventory_data, output_path)
    
    print()
    print("=" * 70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print(f"📁 Output location: {output_path.absolute()}")
    print()
    print("Generated files:")
    print("  1. abc_xyz_matrix.png - Classification heatmaps")
    print("  2. revenue_distribution.png - Revenue analysis by class")
    print("  3. inventory_metrics.png - EOQ, safety stock, reorder points")
    print("  4. cost_analysis.png - Cost breakdown and composition")
    print("  5. service_level_tradeoff.png - Service level optimization")
    print("  6. top_items_dashboard.png - Top 20 items analysis")
    print()


if __name__ == "__main__":
    main()
