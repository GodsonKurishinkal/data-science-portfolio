#!/usr/bin/env python3
"""
Run elasticity analysis on processed pricing data

This script calculates price elasticities for all products in the dataset
and generates summary statistics and visualizations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from src.pricing.elasticity import ElasticityAnalyzer
from src.utils.helpers import load_config, setup_logging, save_results

logger = setup_logging()


def main():
    """Main analysis workflow."""
    logger.info("=" * 60)
    logger.info("PRICE ELASTICITY ANALYSIS")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    elasticity_config = config['elasticity']
    
    # Load processed data
    data_path = project_root / 'data' / 'processed' / 'price_sales_merged.csv'
    logger.info(f"\nLoading data from: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    logger.info(f"Loaded {len(df):,} observations")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Products: {df['item_id'].nunique()}")
    logger.info(f"Stores: {df['store_id'].nunique()}")
    
    # Initialize analyzer
    method = elasticity_config.get('default_method', 'log-log')
    min_obs = elasticity_config.get('min_observations', 30)
    
    logger.info(f"\nInitializing ElasticityAnalyzer")
    logger.info(f"  Method: {method}")
    logger.info(f"  Min observations: {min_obs}")
    
    analyzer = ElasticityAnalyzer(method=method, min_observations=min_obs)
    
    # Calculate elasticities
    logger.info("\nCalculating price elasticities...")
    elasticities = analyzer.calculate_elasticities_batch(
        df=df,
        group_cols=['store_id', 'item_id']
    )
    
    logger.info(f"Calculated elasticities for {len(elasticities)} products")
    logger.info(f"Valid results: {elasticities['valid'].sum()} "
                f"({elasticities['valid'].sum() / len(elasticities) * 100:.1f}%)")
    
    # Filter valid results
    valid_elasticities = elasticities[elasticities['valid'] == True].copy()
    
    if len(valid_elasticities) == 0:
        logger.error("No valid elasticity results. Check data quality.")
        return 1
    
    logger.info("\nValid elasticity statistics:")
    logger.info(f"  Mean: {valid_elasticities['elasticity'].mean():.3f}")
    logger.info(f"  Median: {valid_elasticities['elasticity'].median():.3f}")
    logger.info(f"  Std Dev: {valid_elasticities['elasticity'].std():.3f}")
    logger.info(f"  Range: [{valid_elasticities['elasticity'].min():.3f}, "
                f"{valid_elasticities['elasticity'].max():.3f}]")
    
    logger.info("\nModel fit statistics (R²):")
    logger.info(f"  Mean: {valid_elasticities['r_squared'].mean():.3f}")
    logger.info(f"  Median: {valid_elasticities['r_squared'].median():.3f}")
    logger.info(f"  High quality (R² > 0.3): "
                f"{(valid_elasticities['r_squared'] > 0.3).sum()} "
                f"({(valid_elasticities['r_squared'] > 0.3).sum() / len(valid_elasticities) * 100:.1f}%)")
    
    # Segment by elasticity
    logger.info("\nSegmenting products by elasticity...")
    segmented = analyzer.segment_by_elasticity(valid_elasticities)
    
    logger.info("\nElasticity category distribution:")
    category_counts = segmented['elasticity_category'].value_counts()
    for category, count in category_counts.items():
        pct = count / len(segmented) * 100
        logger.info(f"  {category}: {count} ({pct:.1f}%)")
    
    # Category analysis (Foods, Hobbies, Household)
    segmented['category'] = segmented['item_id'].str.split('_').str[0]
    
    logger.info("\nElasticity by product category:")
    category_stats = segmented.groupby('category')['elasticity'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])
    logger.info("\n" + str(category_stats))
    
    # Get summary
    summary = analyzer.get_elasticity_summary(segmented)
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total products: {summary['total_products']}")
    logger.info(f"Valid results: {summary['valid_results']} "
                f"({summary['valid_percentage']:.1f}%)")
    logger.info(f"Mean elasticity: {summary['mean_elasticity']:.3f}")
    logger.info(f"Median elasticity: {summary['median_elasticity']:.3f}")
    logger.info(f"Mean R²: {summary['mean_r_squared']:.3f}")
    
    # Save results
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save elasticity results
    elasticity_file = output_dir / 'elasticity_results.csv'
    segmented.to_csv(elasticity_file, index=False)
    logger.info(f"\n✓ Saved elasticity results: {elasticity_file}")
    logger.info(f"  Rows: {len(segmented)}, Columns: {len(segmented.columns)}")
    
    # Save summary
    import json
    summary_file = output_dir / 'elasticity_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary: {summary_file}")
    
    # Example: Calculate cross-elasticity for a few products
    logger.info("\nCalculating sample cross-elasticities...")
    foods_items = segmented[segmented['category'] == 'FOODS']['item_id'].unique()[:3]
    
    if len(foods_items) >= 2:
        cross_results = []
        for i, item_a in enumerate(foods_items):
            for item_b in foods_items[i+1:]:
                result = analyzer.calculate_cross_elasticity(
                    product_a_id=item_a,
                    product_b_id=item_b,
                    data=df[df['store_id'] == 'CA_1']
                )
                if result['valid']:
                    cross_results.append(result)
        
        if cross_results:
            cross_df = pd.DataFrame(cross_results)
            cross_file = output_dir / 'cross_elasticity_sample.csv'
            cross_df.to_csv(cross_file, index=False)
            logger.info(f"✓ Saved cross-elasticity sample: {cross_file}")
            logger.info(f"  Relationships: {cross_df['relationship'].value_counts().to_dict()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Review notebook: notebooks/01_price_elasticity_analysis.ipynb")
    logger.info("  2. Proceed to Phase 4: Demand Response Modeling")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Error during elasticity analysis: {e}")
        sys.exit(1)
