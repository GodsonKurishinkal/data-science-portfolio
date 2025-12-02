"""
Integrated Pricing Pipeline

Orchestrates the complete pricing optimization workflow:
1. Elasticity Analysis → Calculate price sensitivity
2. Demand Response → Predict demand at different prices
3. Price Optimization → Find revenue/profit-maximizing prices
4. Markdown Strategy → Plan inventory clearance

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from .elasticity import ElasticityAnalyzer
from .demand_response import DemandResponseModel
from .optimizer import PriceOptimizer
from .markdown import MarkdownOptimizer

logger = logging.getLogger(__name__)


class PricingPipeline:
    """
    End-to-end pricing optimization pipeline.

    This class integrates all pricing modules into a cohesive workflow:
    
    1. **Elasticity Analysis**: Calculate price elasticity for each product
    2. **Demand Modeling**: Predict demand at various price points
    3. **Price Optimization**: Find optimal prices to maximize revenue/profit
    4. **Markdown Planning**: Plan clearance strategies for excess inventory
    
    The pipeline can operate in different modes:
    - `full`: Run complete workflow
    - `elasticity_only`: Calculate elasticities only
    - `optimization_only`: Optimize prices using pre-computed elasticities
    - `markdown_only`: Generate markdown strategies only
    
    Example:
        >>> pipeline = PricingPipeline(objective='revenue')
        >>> results = pipeline.run(
        ...     sales_data=sales_df,
        ...     price_data=prices_df,
        ...     inventory_data=inventory_df
        ... )
        >>> print(f"Revenue lift: {results['summary']['revenue_lift_pct']:.1f}%")
    """

    def __init__(
        self,
        objective: str = 'revenue',
        elasticity_method: str = 'log-log',
        optimization_method: str = 'scipy',
        min_observations: int = 30,
        confidence_level: float = 0.95,
        holding_cost_per_day: float = 0.001,
        salvage_value_pct: float = 0.30
    ):
        """
        Initialize the pricing pipeline with configuration.

        Args:
            objective: Optimization objective ('revenue' or 'profit')
            elasticity_method: Method for elasticity calculation 
                             ('log-log', 'arc', 'point')
            optimization_method: Optimization algorithm 
                               ('scipy', 'grid', 'gradient')
            min_observations: Minimum data points for elasticity calculation
            confidence_level: Confidence level for prediction intervals
            holding_cost_per_day: Daily holding cost rate for markdown
            salvage_value_pct: Salvage value percentage for unsold inventory
        """
        self.objective = objective
        
        # Initialize component modules
        self.elasticity_analyzer = ElasticityAnalyzer(
            method=elasticity_method,
            min_observations=min_observations
        )
        
        self.demand_model = DemandResponseModel(
            elasticity_analyzer=self.elasticity_analyzer,
            use_confidence_intervals=True,
            confidence_level=confidence_level
        )
        
        self.price_optimizer = PriceOptimizer(
            objective=objective,
            method=optimization_method,
            demand_model=self.demand_model
        )
        
        self.markdown_optimizer = MarkdownOptimizer(
            holding_cost_per_day=holding_cost_per_day,
            salvage_value_pct=salvage_value_pct
        )
        
        # Results storage
        self.elasticity_results = None
        self.optimization_results = None
        self.markdown_results = None
        self.pipeline_summary = None
        
        logger.info(
            "Initialized PricingPipeline: objective=%s, elasticity=%s, optimization=%s",
            objective, elasticity_method, optimization_method
        )

    def run(
        self,
        sales_data: pd.DataFrame,
        price_col: str = 'sell_price',
        sales_col: str = 'sales',
        product_col: str = 'item_id',
        store_col: Optional[str] = 'store_id',
        date_col: str = 'date',
        cost_col: Optional[str] = None,
        inventory_data: Optional[pd.DataFrame] = None,
        constraints: Optional[Dict] = None,
        mode: str = 'full'
    ) -> Dict[str, Any]:
        """
        Run the pricing pipeline.

        Args:
            sales_data: DataFrame with historical sales and price data
            price_col: Column name for price
            sales_col: Column name for sales quantity
            product_col: Column name for product identifier
            store_col: Column name for store identifier (optional)
            date_col: Column name for date
            cost_col: Column name for unit cost (optional)
            inventory_data: DataFrame with current inventory levels (optional)
            constraints: Dict of pricing constraints to apply
            mode: Pipeline mode ('full', 'elasticity_only', 'optimization_only')

        Returns:
            Dict containing all pipeline results and summary
        """
        logger.info("Starting pricing pipeline in '%s' mode", mode)
        start_time = datetime.now()
        
        results = {
            'mode': mode,
            'started_at': start_time.isoformat(),
            'config': {
                'objective': self.objective,
                'constraints': constraints or {}
            }
        }
        
        # Step 1: Elasticity Analysis
        if mode in ['full', 'elasticity_only']:
            logger.info("Step 1/4: Calculating elasticities...")
            self.elasticity_results = self._run_elasticity_analysis(
                data=sales_data,
                price_col=price_col,
                sales_col=sales_col,
                product_col=product_col,
                store_col=store_col
            )
            results['elasticity'] = self.elasticity_results
            logger.info("  → Calculated elasticity for %d products", 
                       len(self.elasticity_results['results']))
        
        if mode == 'elasticity_only':
            results['completed_at'] = datetime.now().isoformat()
            return results
        
        # Step 2: Calculate baseline metrics
        logger.info("Step 2/4: Computing baseline metrics...")
        baseline_metrics = self._compute_baseline_metrics(
            data=sales_data,
            price_col=price_col,
            sales_col=sales_col,
            product_col=product_col,
            store_col=store_col,
            cost_col=cost_col
        )
        results['baseline'] = baseline_metrics
        
        # Step 3: Price Optimization
        if mode in ['full', 'optimization_only']:
            logger.info("Step 3/4: Optimizing prices...")
            self.optimization_results = self._run_optimization(
                baseline_metrics=baseline_metrics,
                elasticity_results=self.elasticity_results,
                constraints=constraints,
                cost_col=cost_col
            )
            results['optimization'] = self.optimization_results
            logger.info("  → Optimized prices for %d products",
                       len(self.optimization_results['results']))
        
        # Step 4: Markdown Analysis (if inventory data provided)
        if inventory_data is not None and mode == 'full':
            logger.info("Step 4/4: Generating markdown strategies...")
            self.markdown_results = self._run_markdown_analysis(
                inventory_data=inventory_data,
                baseline_metrics=baseline_metrics,
                elasticity_results=self.elasticity_results,
                product_col=product_col
            )
            results['markdown'] = self.markdown_results
            logger.info("  → Generated markdown strategies for %d products",
                       len(self.markdown_results['results']))
        else:
            logger.info("Step 4/4: Skipping markdown (no inventory data)")
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        results['completed_at'] = datetime.now().isoformat()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("Pipeline completed in %.2f seconds", elapsed)
        
        self.pipeline_summary = results
        return results

    def _run_elasticity_analysis(
        self,
        data: pd.DataFrame,
        price_col: str,
        sales_col: str,
        product_col: str,
        store_col: Optional[str]
    ) -> Dict:
        """Run elasticity analysis on the data."""
        group_cols = [product_col]
        if store_col and store_col in data.columns:
            group_cols = [store_col, product_col]
        
        # Rename columns for elasticity analyzer
        df = data.copy()
        df = df.rename(columns={price_col: 'sell_price', sales_col: 'sales'})
        
        # Calculate elasticities
        elasticities = self.elasticity_analyzer.calculate_elasticities_batch(
            df=df,
            group_cols=group_cols
        )
        
        # Segment by elasticity
        segmented = self.elasticity_analyzer.segment_by_elasticity(elasticities)
        
        # Summary statistics
        summary = self.elasticity_analyzer.get_elasticity_summary(segmented)
        
        return {
            'results': segmented,
            'summary': summary,
            'method': self.elasticity_analyzer.method
        }

    def _compute_baseline_metrics(
        self,
        data: pd.DataFrame,
        price_col: str,
        sales_col: str,
        product_col: str,
        store_col: Optional[str],
        cost_col: Optional[str]
    ) -> pd.DataFrame:
        """Compute baseline demand and revenue metrics."""
        group_cols = [product_col]
        if store_col and store_col in data.columns:
            group_cols.insert(0, store_col)
        
        # Aggregate metrics by product
        agg_dict = {
            sales_col: ['mean', 'std', 'sum'],
            price_col: ['mean', 'min', 'max', 'std']
        }
        
        if cost_col and cost_col in data.columns:
            agg_dict[cost_col] = 'mean'
        
        baseline = data.groupby(group_cols).agg(agg_dict)
        baseline.columns = ['_'.join(col).strip() for col in baseline.columns]
        baseline = baseline.reset_index()
        
        # Rename for clarity
        rename_map = {
            f'{sales_col}_mean': 'baseline_demand',
            f'{sales_col}_std': 'demand_std',
            f'{sales_col}_sum': 'total_sales',
            f'{price_col}_mean': 'current_price',
            f'{price_col}_min': 'min_price_historical',
            f'{price_col}_max': 'max_price_historical',
            f'{price_col}_std': 'price_std'
        }
        if cost_col:
            rename_map[f'{cost_col}_mean'] = 'cost_per_unit'
        
        baseline = baseline.rename(columns=rename_map)
        
        # Calculate revenue
        baseline['baseline_revenue'] = baseline['baseline_demand'] * baseline['current_price']
        
        # Calculate CV for demand variability
        baseline['demand_cv'] = baseline['demand_std'] / baseline['baseline_demand']
        baseline['demand_cv'] = baseline['demand_cv'].fillna(0)
        
        return baseline

    def _run_optimization(
        self,
        baseline_metrics: pd.DataFrame,
        elasticity_results: Dict,
        constraints: Optional[Dict],
        cost_col: Optional[str]
    ) -> Dict:
        """Run price optimization for all products."""
        # Merge elasticity with baseline
        elasticity_df = elasticity_results['results']
        
        # Create product ID from elasticity results
        if 'store_id' in elasticity_df.columns and 'item_id' in elasticity_df.columns:
            elasticity_df['product_key'] = elasticity_df['store_id'].astype(str) + '_' + elasticity_df['item_id'].astype(str)
        elif 'product_id' in elasticity_df.columns:
            elasticity_df['product_key'] = elasticity_df['product_id']
        else:
            elasticity_df['product_key'] = elasticity_df.index.astype(str)
        
        # Create product key in baseline
        if 'store_id' in baseline_metrics.columns and 'item_id' in baseline_metrics.columns:
            baseline_metrics['product_key'] = baseline_metrics['store_id'].astype(str) + '_' + baseline_metrics['item_id'].astype(str)
        elif 'item_id' in baseline_metrics.columns:
            baseline_metrics['product_key'] = baseline_metrics['item_id'].astype(str)
        else:
            baseline_metrics['product_key'] = baseline_metrics.index.astype(str)
        
        # Merge
        merged = baseline_metrics.merge(
            elasticity_df[['product_key', 'elasticity', 'valid', 'elasticity_category']],
            on='product_key',
            how='left'
        )
        
        # Filter to valid elasticities
        valid_products = merged[merged['valid'] == True].copy()
        
        if len(valid_products) == 0:
            logger.warning("No products with valid elasticity estimates")
            return {'results': pd.DataFrame(), 'summary': {}}

        # Prepare for optimization
        products_df = valid_products[
            ['product_key', 'current_price', 'baseline_demand', 'elasticity']
        ].copy()
        products_df = products_df.rename(columns={'product_key': 'product_id'})

        if 'cost_per_unit' in valid_products.columns:
            products_df['cost_per_unit'] = valid_products['cost_per_unit']

        # Set default constraints (only include non-None values)
        default_constraints = {
            'max_discount_pct': 30,
        }
        if cost_col and 'cost_per_unit' in products_df.columns:
            default_constraints['min_margin_pct'] = 10
        if constraints:
            # Only update with non-None constraint values
            for k, v in constraints.items():
                if v is not None:
                    default_constraints[k] = v
        
        # Run portfolio optimization
        results_df = self.price_optimizer.optimize_portfolio(
            products_df=products_df,
            constraints=default_constraints
        )
        
        # Get summary
        opt_summary = self.price_optimizer.get_optimization_summary()
        
        return {
            'results': results_df,
            'summary': opt_summary,
            'constraints_applied': default_constraints
        }

    def _run_markdown_analysis(
        self,
        inventory_data: pd.DataFrame,
        baseline_metrics: pd.DataFrame,
        elasticity_results: Dict,
        product_col: str
    ) -> Dict:
        """Run markdown analysis for products with excess inventory."""
        results = []
        
        # Merge inventory with baseline and elasticity
        elasticity_df = elasticity_results['results']
        
        for _, inv_row in inventory_data.iterrows():
            product_id = inv_row[product_col]
            current_inventory = inv_row.get('inventory', inv_row.get('current_stock', 0))
            days_remaining = inv_row.get('days_remaining', 30)
            
            # Find baseline metrics for this product
            if product_col in baseline_metrics.columns:
                baseline_row = baseline_metrics[baseline_metrics[product_col] == product_id]
            else:
                continue
                
            if len(baseline_row) == 0:
                continue
            
            baseline_row = baseline_row.iloc[0]
            current_price = baseline_row['current_price']
            baseline_demand = baseline_row['baseline_demand']
            
            # Find elasticity
            elasticity = -1.5  # Default
            if 'item_id' in elasticity_df.columns:
                elast_row = elasticity_df[elasticity_df['item_id'] == product_id]
                if len(elast_row) > 0 and elast_row.iloc[0]['valid']:
                    elasticity = elast_row.iloc[0]['elasticity']
            
            # Calculate markdown
            try:
                markdown_result = self.markdown_optimizer.calculate_optimal_markdown(
                    product_id=str(product_id),
                    current_inventory=int(current_inventory),
                    days_remaining=int(days_remaining),
                    current_price=float(current_price),
                    elasticity=float(elasticity),
                    baseline_demand=float(baseline_demand),
                    cost_per_unit=baseline_row.get('cost_per_unit')
                )
                results.append(markdown_result)
            except (ValueError, KeyError) as e:
                logger.warning("Markdown failed for %s: %s", product_id, str(e))
                continue
        
        return {
            'results': results,
            'summary': {
                'products_analyzed': len(results),
                'avg_clearance_rate': np.mean([r['clearance_rate'] for r in results]) if results else 0,
                'total_expected_revenue': sum(r['expected_revenue'] for r in results) if results else 0
            }
        }

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate overall pipeline summary."""
        summary = {
            'mode': results['mode'],
            'objective': self.objective
        }
        
        # Elasticity summary
        if 'elasticity' in results:
            elast = results['elasticity']['summary']
            summary['elasticity'] = {
                'total_products': elast.get('total_products', 0),
                'valid_estimates': elast.get('valid_results', 0),
                'avg_elasticity': round(elast.get('mean_elasticity', 0), 3),
                'median_elasticity': round(elast.get('median_elasticity', 0), 3)
            }
        
        # Optimization summary
        if 'optimization' in results and len(results['optimization']['results']) > 0:
            opt_df = results['optimization']['results']
            opt_summary = results['optimization']['summary']
            
            summary['optimization'] = {
                'products_optimized': len(opt_df),
                'avg_price_change_pct': opt_summary.get('avg_price_change_pct', 0),
                'avg_revenue_lift_pct': opt_summary.get('avg_revenue_change_pct', 0),
                'total_current_revenue': opt_summary.get('total_current_revenue', 0),
                'total_optimal_revenue': opt_summary.get('total_optimal_revenue', 0),
                'total_revenue_gain': opt_summary.get('total_revenue_gain', 0)
            }
            
            # Business impact
            if summary['optimization']['total_current_revenue'] > 0:
                lift_pct = (
                    summary['optimization']['total_revenue_gain'] / 
                    summary['optimization']['total_current_revenue'] * 100
                )
                summary['revenue_lift_pct'] = round(lift_pct, 2)
        
        # Markdown summary
        if 'markdown' in results:
            md_summary = results['markdown']['summary']
            summary['markdown'] = {
                'products_analyzed': md_summary.get('products_analyzed', 0),
                'avg_clearance_rate': round(md_summary.get('avg_clearance_rate', 0) * 100, 1),
                'total_clearance_revenue': md_summary.get('total_expected_revenue', 0)
            }
        
        return summary

    def get_recommendations(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top pricing recommendations from optimization results.

        Args:
            top_n: Number of top recommendations to return

        Returns:
            DataFrame with prioritized pricing recommendations
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results. Run pipeline first.")
        
        results = self.optimization_results['results'].copy()
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Sort by revenue gain potential
        results['revenue_gain'] = results['optimal_revenue'] - results['current_revenue']
        results = results.sort_values('revenue_gain', ascending=False)
        
        # Select top recommendations
        recommendations = results.head(top_n)[[
            'product_id',
            'current_price',
            'optimal_price',
            'price_change_pct',
            'current_revenue',
            'optimal_revenue',
            'revenue_change_pct',
            'elasticity'
        ]].copy()
        
        # Add action recommendation
        def get_action(row):
            if row['price_change_pct'] > 5:
                return '📈 Increase price'
            elif row['price_change_pct'] < -5:
                return '📉 Decrease price'
            else:
                return '➡️ Maintain price'
        
        recommendations['action'] = recommendations.apply(get_action, axis=1)
        
        return recommendations

    def generate_report(self) -> str:
        """
        Generate a text summary report of pipeline results.

        Returns:
            Formatted text report
        """
        if self.pipeline_summary is None:
            return "No results available. Run pipeline first."
        
        summary = self.pipeline_summary['summary']
        
        lines = [
            "=" * 60,
            "DYNAMIC PRICING ENGINE - PIPELINE REPORT",
            "=" * 60,
            "",
            f"Mode: {summary['mode'].upper()}",
            f"Objective: {summary['objective'].upper()} Maximization",
            ""
        ]
        
        if 'elasticity' in summary:
            e = summary['elasticity']
            lines.extend([
                "📊 ELASTICITY ANALYSIS",
                "-" * 40,
                f"  Products analyzed: {e['total_products']}",
                f"  Valid estimates: {e['valid_estimates']} ({e['valid_estimates']/e['total_products']*100:.0f}%)",
                f"  Average elasticity: {e['avg_elasticity']:.3f}",
                f"  Median elasticity: {e['median_elasticity']:.3f}",
                ""
            ])
        
        if 'optimization' in summary:
            o = summary['optimization']
            lines.extend([
                "💰 PRICE OPTIMIZATION",
                "-" * 40,
                f"  Products optimized: {o['products_optimized']}",
                f"  Avg price change: {o['avg_price_change_pct']:+.1f}%",
                f"  Current revenue: ${o['total_current_revenue']:,.2f}",
                f"  Optimal revenue: ${o['total_optimal_revenue']:,.2f}",
                f"  Revenue gain: ${o['total_revenue_gain']:,.2f} ({summary.get('revenue_lift_pct', 0):+.1f}%)",
                ""
            ])
        
        if 'markdown' in summary:
            m = summary['markdown']
            lines.extend([
                "🔻 MARKDOWN STRATEGY",
                "-" * 40,
                f"  Products analyzed: {m['products_analyzed']}",
                f"  Avg clearance rate: {m['avg_clearance_rate']:.1f}%",
                f"  Expected clearance revenue: ${m['total_clearance_revenue']:,.2f}",
                ""
            ])
        
        lines.extend([
            "=" * 60,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ])
        
        return "\n".join(lines)
