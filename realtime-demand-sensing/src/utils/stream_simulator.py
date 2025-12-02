"""
Stream Simulator - Data Generation for Real-Time Sensing

Generates realistic streaming sales data with patterns:
- Daily seasonality (peak during business hours)
- Weekly seasonality (weekends vs weekdays)
- Random noise and occasional anomalies
- Inventory tracking with consumption

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Generator
import logging

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Simulate streaming sales data for real-time demand sensing.
    
    Generates hourly sales data with realistic patterns including:
    - Intraday patterns (peak hours, quiet periods)
    - Weekly patterns (weekends vs weekdays)
    - Random noise
    - Occasional anomalies (spikes and drops)
    
    Example:
        >>> simulator = StreamSimulator(n_products=10)
        >>> historical = simulator.generate_historical(days=60)
        >>> for batch in simulator.stream(interval_hours=1):
        ...     process(batch)
    """
    
    def __init__(
        self,
        n_products: int = 10,
        base_demand_range: Tuple[float, float] = (50, 200),
        noise_level: float = 0.15,
        anomaly_probability: float = 0.02,
        seed: Optional[int] = 42
    ):
        """
        Initialize the stream simulator.
        
        Args:
            n_products: Number of products to simulate
            base_demand_range: (min, max) range for base hourly demand
            noise_level: Standard deviation as fraction of demand
            anomaly_probability: Probability of anomaly per hour per product
            seed: Random seed for reproducibility
        """
        self.n_products = n_products
        self.base_demand_range = base_demand_range
        self.noise_level = noise_level
        self.anomaly_probability = anomaly_probability
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate product characteristics
        self.products = self._generate_products()
        
        # Track state
        self.current_timestamp = datetime.now()
        self.inventory_state: Dict[str, float] = {}
        
        logger.info(
            "Initialized StreamSimulator: %d products, noise=%.2f, anomaly_prob=%.3f",
            n_products, noise_level, anomaly_probability
        )
    
    def _generate_products(self) -> pd.DataFrame:
        """Generate product master data with characteristics."""
        products = []
        
        for i in range(self.n_products):
            product_id = f"PROD_{i+1:03d}"
            
            # Random characteristics
            base_demand = np.random.uniform(*self.base_demand_range)
            
            # Category affects patterns
            category = np.random.choice(
                ['GROCERY', 'ELECTRONICS', 'APPAREL', 'HOME'],
                p=[0.4, 0.2, 0.25, 0.15]
            )
            
            # Category-specific patterns
            if category == 'GROCERY':
                weekend_factor = 1.3
                peak_hour = 18  # Evening
            elif category == 'ELECTRONICS':
                weekend_factor = 1.5
                peak_hour = 14  # Afternoon
            elif category == 'APPAREL':
                weekend_factor = 1.6
                peak_hour = 15  # Afternoon
            else:  # HOME
                weekend_factor = 1.4
                peak_hour = 11  # Morning
            
            products.append({
                'product_id': product_id,
                'category': category,
                'base_demand': base_demand,
                'weekend_factor': weekend_factor,
                'peak_hour': peak_hour,
                'unit_cost': np.random.uniform(5, 100),
                'lead_time_days': np.random.choice([2, 3, 4, 5, 7]),
                'initial_inventory': base_demand * np.random.uniform(20, 40)
            })
        
        return pd.DataFrame(products)
    
    def _hourly_pattern(self, hour: int, peak_hour: int) -> float:
        """
        Calculate intraday demand pattern.
        
        Uses a sinusoidal pattern centered on peak hour.
        """
        # Distance from peak hour (circular)
        dist = min(abs(hour - peak_hour), 24 - abs(hour - peak_hour))
        
        # Sinusoidal pattern with minimum at 4 AM
        amplitude = 0.4
        baseline = 0.7
        
        # Pattern peaks at peak_hour, minimum at 4 AM
        pattern = baseline + amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24)
        
        # Extra reduction for very late night / early morning
        if 2 <= hour <= 6:
            pattern *= 0.5
        
        return max(0.1, pattern)
    
    def _weekly_pattern(self, day_of_week: int, weekend_factor: float) -> float:
        """
        Calculate weekly demand pattern.
        
        Args:
            day_of_week: 0=Monday, 6=Sunday
            weekend_factor: Multiplier for weekend days
        """
        if day_of_week in [5, 6]:  # Weekend
            return weekend_factor
        elif day_of_week == 4:  # Friday
            return 1.0 + (weekend_factor - 1.0) * 0.5
        else:  # Weekday
            return 1.0
    
    def _generate_anomaly(self) -> Tuple[bool, float]:
        """
        Determine if an anomaly occurs and its magnitude.
        
        Returns:
            Tuple of (is_anomaly, multiplier)
        """
        if np.random.random() < self.anomaly_probability:
            # 60% spikes, 40% drops
            if np.random.random() < 0.6:
                # Spike: 2x to 4x normal
                multiplier = np.random.uniform(2.0, 4.0)
            else:
                # Drop: 0.1x to 0.4x normal
                multiplier = np.random.uniform(0.1, 0.4)
            return True, multiplier
        return False, 1.0
    
    def generate_hourly_sales(
        self,
        timestamp: datetime,
        product_row: pd.Series
    ) -> Dict:
        """
        Generate sales for a single product at a specific hour.
        
        Args:
            timestamp: The datetime for this observation
            product_row: Product characteristics from products DataFrame
        
        Returns:
            Dict with sales data
        """
        hour = timestamp.hour
        dow = timestamp.weekday()
        
        # Base demand with patterns
        base = product_row['base_demand']
        hourly_mult = self._hourly_pattern(hour, product_row['peak_hour'])
        weekly_mult = self._weekly_pattern(dow, product_row['weekend_factor'])
        
        # Expected demand
        expected_demand = base * hourly_mult * weekly_mult
        
        # Add noise
        noise = np.random.normal(0, expected_demand * self.noise_level)
        
        # Check for anomaly
        is_anomaly, anomaly_mult = self._generate_anomaly()
        
        # Final sales
        sales = max(0, (expected_demand + noise) * anomaly_mult)
        
        return {
            'timestamp': timestamp,
            'product_id': product_row['product_id'],
            'category': product_row['category'],
            'sales': round(sales, 1),
            'expected_sales': round(expected_demand, 1),
            'is_anomaly': is_anomaly,
            'anomaly_type': 'spike' if is_anomaly and anomaly_mult > 1 else 'drop' if is_anomaly else None,
            'hour': hour,
            'day_of_week': dow,
            'is_weekend': dow in [5, 6]
        }
    
    def generate_historical(
        self,
        days: int = 60,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate historical data for training and baseline calculation.
        
        Args:
            days: Number of days of history
            end_date: End date (defaults to now)
        
        Returns:
            DataFrame with hourly sales data
        """
        if end_date is None:
            end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        start_date = end_date - timedelta(days=days)
        
        logger.info("Generating %d days of historical data: %s to %s", 
                   days, start_date, end_date)
        
        records = []
        current_time = start_date
        
        while current_time < end_date:
            for _, product in self.products.iterrows():
                record = self.generate_hourly_sales(current_time, product)
                records.append(record)
            
            current_time += timedelta(hours=1)
        
        df = pd.DataFrame(records)
        
        # Add inventory tracking
        df = self._add_inventory_tracking(df)
        
        logger.info("Generated %d records for %d products", 
                   len(df), self.n_products)
        
        return df
    
    def _add_inventory_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inventory levels based on sales consumption."""
        df = df.copy()
        df['inventory'] = 0.0
        
        for product_id in df['product_id'].unique():
            mask = df['product_id'] == product_id
            product = self.products[self.products['product_id'] == product_id].iloc[0]
            
            initial_inv = product['initial_inventory']
            
            # Calculate cumulative consumption
            cumulative_sales = df.loc[mask, 'sales'].cumsum()
            
            # Replenishment simulation (when inventory drops below 20% of initial)
            inventory = []
            current_inv = initial_inv
            
            for sale in df.loc[mask, 'sales']:
                current_inv -= sale
                
                # Trigger replenishment when low
                if current_inv < initial_inv * 0.2:
                    current_inv += initial_inv * 0.8  # Replenish
                
                inventory.append(max(0, current_inv))
            
            df.loc[mask, 'inventory'] = inventory
        
        return df
    
    def stream(
        self,
        interval_hours: int = 1,
        max_iterations: Optional[int] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate streaming data as a generator.
        
        Args:
            interval_hours: Hours between batches
            max_iterations: Maximum number of batches (None = infinite)
        
        Yields:
            DataFrame with one batch of hourly data for all products
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            records = []
            
            for _, product in self.products.iterrows():
                record = self.generate_hourly_sales(self.current_timestamp, product)
                
                # Update inventory state
                product_id = product['product_id']
                if product_id not in self.inventory_state:
                    self.inventory_state[product_id] = product['initial_inventory']
                
                self.inventory_state[product_id] -= record['sales']
                
                # Replenishment check
                if self.inventory_state[product_id] < product['initial_inventory'] * 0.2:
                    self.inventory_state[product_id] += product['initial_inventory'] * 0.8
                
                record['inventory'] = max(0, self.inventory_state[product_id])
                records.append(record)
            
            yield pd.DataFrame(records)
            
            self.current_timestamp += timedelta(hours=interval_hours)
            iteration += 1
    
    def get_products(self) -> pd.DataFrame:
        """Get product master data."""
        return self.products.copy()
    
    def get_product_summary(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate product-level summary statistics.
        
        Args:
            historical_data: Historical sales DataFrame
        
        Returns:
            DataFrame with product summary
        """
        summary = historical_data.groupby('product_id').agg({
            'sales': ['mean', 'std', 'sum', 'min', 'max'],
            'is_anomaly': 'sum',
            'inventory': 'last'
        }).round(2)
        
        summary.columns = ['avg_hourly_sales', 'std_sales', 'total_sales',
                          'min_sales', 'max_sales', 'anomaly_count', 'current_inventory']
        
        summary['daily_sales'] = summary['avg_hourly_sales'] * 24
        summary['cv'] = summary['std_sales'] / summary['avg_hourly_sales']
        summary['days_of_stock'] = summary['current_inventory'] / summary['daily_sales']
        
        # Merge with product master
        summary = summary.reset_index().merge(
            self.products[['product_id', 'category', 'unit_cost', 'lead_time_days']],
            on='product_id'
        )
        
        return summary


def create_demo_data(
    n_products: int = 20,
    days: int = 60,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create demo data.
    
    Args:
        n_products: Number of products
        days: Days of history
        seed: Random seed
    
    Returns:
        Tuple of (historical_data, product_summary)
    """
    simulator = StreamSimulator(n_products=n_products, seed=seed)
    historical = simulator.generate_historical(days=days)
    summary = simulator.get_product_summary(historical)
    
    return historical, summary


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    simulator = StreamSimulator(n_products=5, seed=42)
    data = simulator.generate_historical(days=7)
    
    print(f"\nGenerated {len(data)} records")
    print(f"\nSample data:")
    print(data.head(10))
    
    print(f"\nProduct summary:")
    summary = simulator.get_product_summary(data)
    print(summary)
