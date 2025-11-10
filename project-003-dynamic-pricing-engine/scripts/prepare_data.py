"""
Data preparation script

Loads M5 data, preprocesses it, and saves the pricing dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.loader import PricingDataLoader
from src.data.preprocessing import PricingDataPreprocessor
from src.utils.helpers import setup_logging, load_config
import logging

# Setup logging
logger = setup_logging()


def main():
    """Prepare pricing dataset."""
    print("=" * 80)
    print("📊 DATA PREPARATION - Dynamic Pricing Engine")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    logger.info("Initializing data loader")
    loader = PricingDataLoader(data_path='data/raw')
    
    # Load sample data for demo (faster processing)
    print("📥 Loading M5 data...")
    print("   - Stores: CA_1")
    print("   - Items: 100 sampled items")
    print("   - Date range: 2015-01-01 to 2016-06-19")
    print()
    
    df = loader.load_sample_for_demo()
    
    print(f"✅ Data loaded: {df.shape}")
    print(f"   - Unique items: {df['item_id'].nunique()}")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Observations: {len(df):,}")
    print()
    
    # Initialize preprocessor
    logger.info("Initializing data preprocessor")
    preprocessor = PricingDataPreprocessor()
    
    # Create pricing dataset
    print("🔧 Preprocessing data...")
    print("   - Extracting price history")
    print("   - Calculating price statistics")
    print("   - Identifying promotions")
    print("   - Engineering pricing features")
    print()
    
    df_processed = preprocessor.create_pricing_dataset(df, include_all_features=True)
    
    print(f"✅ Preprocessing complete: {df_processed.shape}")
    print(f"   - Total features: {len(df_processed.columns)}")
    print()
    
    # Display sample statistics
    print("📈 Price Statistics:")
    print(f"   - Average price: ${df_processed['sell_price'].mean():.2f}")
    print(f"   - Price range: ${df_processed['sell_price'].min():.2f} - ${df_processed['sell_price'].max():.2f}")
    print(f"   - Price changes: {df_processed['price_changed'].sum():,}")
    print(f"   - Promotional days: {df_processed['is_promotion'].sum():,} ({(df_processed['is_promotion'].sum() / len(df_processed) * 100):.1f}%)")
    print()
    
    # Save processed data
    output_path = Path('data/processed/price_sales_merged.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving processed data to {output_path}")
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    
    # Create product summary
    print("📊 Creating product price summary...")
    summary = preprocessor.get_product_price_summary(df_processed)
    
    summary_path = Path('data/processed/product_price_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")
    print(f"   - Products: {len(summary)}")
    print()
    
    # Display sample of processed data
    print("📋 Sample of processed data (first 3 rows):")
    print(df_processed[['date', 'item_id', 'store_id', 'sell_price', 'sales', 
                        'price_change_pct', 'is_promotion', 'price_vs_mean']].head(3).to_string())
    print()
    
    print("=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - Phase 3: Implement price elasticity analysis")
    print("  - Use processed data for elasticity calculations")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
