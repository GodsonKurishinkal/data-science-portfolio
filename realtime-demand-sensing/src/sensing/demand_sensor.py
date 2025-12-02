"""
Demand Sensor - Real-Time Demand Estimation

Provides real-time demand estimation using:
- Signal fusion (combining multiple data sources)
- Exponential smoothing with drift detection
- Baseline tracking and comparison

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DemandSensor:
    """
    Real-time demand sensing with signal fusion and drift detection.
    
    This class provides:
    - Exponential smoothing for demand estimation
    - Drift detection when demand deviates from baseline
    - Rolling statistics for baseline calculation
    - Signal fusion from multiple sources
    
    Example:
        >>> sensor = DemandSensor(alpha=0.3, drift_threshold=2.0)
        >>> sensor.initialize_baseline(historical_data)
        >>> 
        >>> # Process new observations
        >>> result = sensor.update(product_id='PROD_001', sales=150)
        >>> print(f"Demand: {result['smoothed_demand']:.1f}, Drift: {result['is_drift']}")
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        drift_threshold: float = 2.0,
        lookback_hours: int = 168,  # 7 days
        min_observations: int = 24
    ):
        """
        Initialize the demand sensor.
        
        Args:
            alpha: Exponential smoothing parameter (0 < alpha <= 1)
            drift_threshold: Z-score threshold for drift detection
            lookback_hours: Hours of history for baseline calculation
            min_observations: Minimum observations before drift detection
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.drift_threshold = drift_threshold
        self.lookback_hours = lookback_hours
        self.min_observations = min_observations
        
        # State per product
        self.smoothed_values: Dict[str, float] = {}
        self.baselines: Dict[str, Dict] = {}
        self.history: Dict[str, deque] = {}
        self.observation_counts: Dict[str, int] = {}
        
        logger.info(
            "Initialized DemandSensor: alpha=%.2f, drift_threshold=%.1f",
            alpha, drift_threshold
        )
    
    def initialize_baseline(
        self,
        historical_data: pd.DataFrame,
        product_col: str = 'product_id',
        sales_col: str = 'sales',
        timestamp_col: str = 'timestamp'
    ) -> None:
        """
        Initialize baselines from historical data.
        
        Args:
            historical_data: DataFrame with historical sales
            product_col: Column name for product ID
            sales_col: Column name for sales quantity
            timestamp_col: Column name for timestamp
        """
        for product_id in historical_data[product_col].unique():
            product_data = historical_data[
                historical_data[product_col] == product_id
            ].sort_values(timestamp_col)
            
            sales = product_data[sales_col].values
            
            # Calculate baseline statistics
            self.baselines[product_id] = {
                'mean': float(np.mean(sales)),
                'std': float(np.std(sales)),
                'median': float(np.median(sales)),
                'min': float(np.min(sales)),
                'max': float(np.max(sales)),
                'last_updated': datetime.now()
            }
            
            # Initialize smoothed value with last observation
            self.smoothed_values[product_id] = float(sales[-1])
            
            # Initialize history
            recent_sales = sales[-self.lookback_hours:] if len(sales) > self.lookback_hours else sales
            self.history[product_id] = deque(recent_sales, maxlen=self.lookback_hours)
            
            # Track observations
            self.observation_counts[product_id] = len(sales)
        
        logger.info("Initialized baselines for %d products", len(self.baselines))
    
    def update(
        self,
        product_id: str,
        sales: float,
        timestamp: Optional[datetime] = None,
        external_signals: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Update demand estimate with new observation.
        
        Args:
            product_id: Product identifier
            sales: Sales quantity observed
            timestamp: Observation timestamp (defaults to now)
            external_signals: Optional dict of external signal values
        
        Returns:
            Dict with sensing results:
                - smoothed_demand: Exponentially smoothed demand
                - baseline_demand: Historical baseline
                - demand_change_pct: Percent change from baseline
                - z_score: Standard deviations from baseline
                - is_drift: Whether drift was detected
                - drift_direction: 'up', 'down', or None
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize if new product
        if product_id not in self.smoothed_values:
            self.smoothed_values[product_id] = sales
            self.baselines[product_id] = {
                'mean': sales,
                'std': 1.0,
                'median': sales,
                'min': sales,
                'max': sales,
                'last_updated': timestamp
            }
            self.history[product_id] = deque([sales], maxlen=self.lookback_hours)
            self.observation_counts[product_id] = 1
        
        # Exponential smoothing
        prev_smoothed = self.smoothed_values[product_id]
        new_smoothed = self.alpha * sales + (1 - self.alpha) * prev_smoothed
        self.smoothed_values[product_id] = new_smoothed
        
        # Update history
        self.history[product_id].append(sales)
        self.observation_counts[product_id] += 1
        
        # Get baseline
        baseline = self.baselines[product_id]
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        # Calculate metrics
        demand_change = new_smoothed - baseline_mean
        demand_change_pct = (demand_change / baseline_mean * 100) if baseline_mean > 0 else 0
        
        # Z-score calculation
        z_score = demand_change / baseline_std if baseline_std > 0 else 0
        
        # Drift detection (only after sufficient observations)
        is_drift = False
        drift_direction = None
        
        if self.observation_counts[product_id] >= self.min_observations:
            if abs(z_score) > self.drift_threshold:
                is_drift = True
                drift_direction = 'up' if z_score > 0 else 'down'
        
        # Apply signal fusion if external signals provided
        fused_demand = new_smoothed
        if external_signals:
            fused_demand = self._fuse_signals(new_smoothed, external_signals)
        
        result = {
            'product_id': product_id,
            'timestamp': timestamp,
            'raw_sales': sales,
            'smoothed_demand': round(new_smoothed, 2),
            'fused_demand': round(fused_demand, 2),
            'baseline_demand': round(baseline_mean, 2),
            'baseline_std': round(baseline_std, 2),
            'demand_change': round(demand_change, 2),
            'demand_change_pct': round(demand_change_pct, 2),
            'z_score': round(z_score, 2),
            'is_drift': is_drift,
            'drift_direction': drift_direction,
            'observations': self.observation_counts[product_id]
        }
        
        if is_drift:
            logger.info(
                "Drift detected for %s: %.1f%% change (z=%.2f)",
                product_id, demand_change_pct, z_score
            )
        
        return result
    
    def _fuse_signals(
        self,
        sales_demand: float,
        external_signals: Dict[str, float]
    ) -> float:
        """
        Fuse multiple demand signals with weighted average.
        
        Args:
            sales_demand: Primary demand from sales
            external_signals: Dict of signal_name -> value
        
        Returns:
            Fused demand estimate
        """
        # Default weights
        weights = {
            'sales': 0.60,
            'web_traffic': 0.15,
            'inventory_velocity': 0.15,
            'pos_scans': 0.10
        }
        
        total_weight = weights['sales']
        weighted_sum = sales_demand * weights['sales']
        
        for signal, value in external_signals.items():
            if signal in weights:
                weighted_sum += value * weights[signal]
                total_weight += weights[signal]
        
        return weighted_sum / total_weight if total_weight > 0 else sales_demand
    
    def update_baseline(
        self,
        product_id: str,
        recalculate: bool = True
    ) -> Dict:
        """
        Update baseline from recent history.
        
        Args:
            product_id: Product to update
            recalculate: Whether to recalculate from history
        
        Returns:
            Updated baseline dict
        """
        if product_id not in self.history:
            raise ValueError(f"No history for product {product_id}")
        
        if recalculate:
            history_array = np.array(self.history[product_id])
            
            self.baselines[product_id] = {
                'mean': float(np.mean(history_array)),
                'std': float(np.std(history_array)),
                'median': float(np.median(history_array)),
                'min': float(np.min(history_array)),
                'max': float(np.max(history_array)),
                'last_updated': datetime.now()
            }
        
        return self.baselines[product_id]
    
    def get_current_demand(self, product_id: str) -> Optional[float]:
        """Get current smoothed demand for a product."""
        return self.smoothed_values.get(product_id)
    
    def get_baseline(self, product_id: str) -> Optional[Dict]:
        """Get baseline statistics for a product."""
        return self.baselines.get(product_id)
    
    def get_all_demands(self) -> Dict[str, float]:
        """Get current demand for all products."""
        return self.smoothed_values.copy()
    
    def get_status_summary(self) -> pd.DataFrame:
        """
        Get summary status for all products.
        
        Returns:
            DataFrame with product status
        """
        records = []
        
        for product_id in self.smoothed_values.keys():
            baseline = self.baselines.get(product_id, {})
            smoothed = self.smoothed_values.get(product_id, 0)
            
            baseline_mean = baseline.get('mean', smoothed)
            baseline_std = baseline.get('std', 1)
            
            change_pct = ((smoothed - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            z_score = (smoothed - baseline_mean) / baseline_std if baseline_std > 0 else 0
            
            records.append({
                'product_id': product_id,
                'current_demand': round(smoothed, 2),
                'baseline_demand': round(baseline_mean, 2),
                'change_pct': round(change_pct, 2),
                'z_score': round(z_score, 2),
                'status': 'DRIFT' if abs(z_score) > self.drift_threshold else 'NORMAL',
                'observations': self.observation_counts.get(product_id, 0)
            })
        
        return pd.DataFrame(records)


class DemandSensorBatch:
    """
    Batch demand sensing for multiple products.
    
    Wraps DemandSensor for efficient batch processing.
    """
    
    def __init__(self, **sensor_kwargs):
        """Initialize with DemandSensor parameters."""
        self.sensor = DemandSensor(**sensor_kwargs)
    
    def initialize(self, historical_data: pd.DataFrame, **kwargs) -> None:
        """Initialize from historical data."""
        self.sensor.initialize_baseline(historical_data, **kwargs)
    
    def update_batch(
        self,
        batch_data: pd.DataFrame,
        product_col: str = 'product_id',
        sales_col: str = 'sales',
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Process a batch of observations.
        
        Args:
            batch_data: DataFrame with new observations
            product_col: Column for product ID
            sales_col: Column for sales
            timestamp_col: Column for timestamp
        
        Returns:
            DataFrame with sensing results
        """
        results = []
        
        for _, row in batch_data.iterrows():
            result = self.sensor.update(
                product_id=row[product_col],
                sales=row[sales_col],
                timestamp=row.get(timestamp_col, datetime.now())
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_drift_alerts(self) -> pd.DataFrame:
        """Get products with detected drift."""
        status = self.sensor.get_status_summary()
        return status[status['status'] == 'DRIFT']


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    from stream_simulator import StreamSimulator
    
    # Generate test data
    simulator = StreamSimulator(n_products=5, seed=42)
    historical = simulator.generate_historical(days=30)
    
    # Initialize sensor
    sensor = DemandSensor(alpha=0.3, drift_threshold=2.0)
    sensor.initialize_baseline(historical)
    
    # Simulate streaming updates
    print("\nProcessing new observations...")
    
    stream_gen = simulator.stream(interval_hours=1, max_iterations=5)
    for batch in stream_gen:
        for _, row in batch.iterrows():
            result = sensor.update(row['product_id'], row['sales'])
            if result['is_drift']:
                print(f"  DRIFT: {row['product_id']} - {result['demand_change_pct']:+.1f}%")
    
    print("\nStatus summary:")
    print(sensor.get_status_summary())
