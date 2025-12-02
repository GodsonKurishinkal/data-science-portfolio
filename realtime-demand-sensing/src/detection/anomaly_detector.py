"""
Anomaly Detector - Multi-Method Anomaly Detection

Provides ensemble anomaly detection using:
- Statistical methods (Z-score, IQR)
- Machine Learning (Isolation Forest)
- Business rules (stockout risk, excess inventory)

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    STOCKOUT_RISK = "stockout_risk"
    EXCESS_INVENTORY = "excess_inventory"
    PATTERN_BREAK = "pattern_break"


@dataclass
class Anomaly:
    """Anomaly detection result."""
    product_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    value: float
    expected_value: float
    deviation: float
    z_score: float
    message: str
    detection_method: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'product_id': self.product_id,
            'timestamp': self.timestamp,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation': self.deviation,
            'z_score': self.z_score,
            'message': self.message,
            'detection_method': self.detection_method,
            'metadata': self.metadata or {}
        }


class ZScoreDetector:
    """Z-score based anomaly detection."""
    
    def __init__(
        self,
        threshold: float = 3.0,
        window_size: int = 168  # 7 days hourly
    ):
        """
        Initialize Z-score detector.
        
        Args:
            threshold: Z-score threshold for anomaly
            window_size: Rolling window for statistics
        """
        self.threshold = threshold
        self.window_size = window_size
    
    def detect(
        self,
        data: pd.Series,
        product_id: str = "unknown"
    ) -> List[Anomaly]:
        """
        Detect anomalies using Z-score.
        
        Args:
            data: Time series of values
            product_id: Product identifier
        
        Returns:
            List of detected anomalies
        """
        if len(data) < 2:
            return []
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(
            window=min(self.window_size, len(data)),
            min_periods=1
        ).mean()
        
        rolling_std = data.rolling(
            window=min(self.window_size, len(data)),
            min_periods=1
        ).std()
        
        # Handle zero std
        rolling_std = rolling_std.replace(0, 1)
        
        # Calculate Z-scores
        z_scores = (data - rolling_mean) / rolling_std
        
        # Find anomalies
        anomalies = []
        anomaly_mask = np.abs(z_scores) > self.threshold
        
        for idx in data.index[anomaly_mask]:
            z = z_scores[idx]
            value = data[idx]
            expected = rolling_mean[idx]
            
            # Determine type and severity
            if z > 0:
                anomaly_type = AnomalyType.DEMAND_SPIKE
            else:
                anomaly_type = AnomalyType.DEMAND_DROP
            
            if abs(z) > self.threshold * 2:
                severity = AnomalySeverity.CRITICAL
            elif abs(z) > self.threshold * 1.5:
                severity = AnomalySeverity.WARNING
            else:
                severity = AnomalySeverity.INFO
            
            anomaly = Anomaly(
                product_id=product_id,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                value=float(value),
                expected_value=float(expected),
                deviation=float(value - expected),
                z_score=float(z),
                message=f"{'Spike' if z > 0 else 'Drop'} detected: {value:.1f} vs expected {expected:.1f} (z={z:.2f})",
                detection_method="z_score"
            )
            anomalies.append(anomaly)
        
        return anomalies


class IQRDetector:
    """Interquartile Range based anomaly detection."""
    
    def __init__(
        self,
        multiplier: float = 1.5,
        window_size: int = 168
    ):
        """
        Initialize IQR detector.
        
        Args:
            multiplier: IQR multiplier for bounds
            window_size: Rolling window for statistics
        """
        self.multiplier = multiplier
        self.window_size = window_size
    
    def detect(
        self,
        data: pd.Series,
        product_id: str = "unknown"
    ) -> List[Anomaly]:
        """
        Detect anomalies using IQR.
        
        Args:
            data: Time series of values
            product_id: Product identifier
        
        Returns:
            List of detected anomalies
        """
        if len(data) < 4:
            return []
        
        # Calculate IQR bounds
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr
        
        # Find anomalies
        anomalies = []
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        
        median = data.median()
        std = data.std()
        
        for idx in data.index[anomaly_mask]:
            value = data[idx]
            z = (value - median) / std if std > 0 else 0
            
            if value > upper_bound:
                anomaly_type = AnomalyType.DEMAND_SPIKE
            else:
                anomaly_type = AnomalyType.DEMAND_DROP
            
            severity = AnomalySeverity.WARNING
            if value > upper_bound * 1.5 or value < lower_bound * 0.5:
                severity = AnomalySeverity.CRITICAL
            
            anomaly = Anomaly(
                product_id=product_id,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                value=float(value),
                expected_value=float(median),
                deviation=float(value - median),
                z_score=float(z),
                message=f"IQR outlier: {value:.1f} outside [{lower_bound:.1f}, {upper_bound:.1f}]",
                detection_method="iqr"
            )
            anomalies.append(anomaly)
        
        return anomalies


class IsolationForestDetector:
    """Isolation Forest based anomaly detection."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for IsolationForestDetector")
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the Isolation Forest model.
        
        Args:
            data: Training data (features as columns)
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(data)
        logger.info("Fitted IsolationForest on %d samples", len(data))
    
    def detect(
        self,
        data: pd.DataFrame,
        product_id: str = "unknown",
        value_col: str = 'value'
    ) -> List[Anomaly]:
        """
        Detect anomalies using fitted model.
        
        Args:
            data: Data to check for anomalies
            product_id: Product identifier
            value_col: Column containing main value
        
        Returns:
            List of detected anomalies
        """
        if self.model is None:
            self.fit(data)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(data)
        scores = self.model.score_samples(data)
        
        anomalies = []
        anomaly_indices = np.where(predictions == -1)[0]
        
        mean_val = data[value_col].mean() if value_col in data.columns else data.iloc[:, 0].mean()
        std_val = data[value_col].std() if value_col in data.columns else data.iloc[:, 0].std()
        
        for idx in anomaly_indices:
            row = data.iloc[idx]
            value = row[value_col] if value_col in data.columns else row.iloc[0]
            z = (value - mean_val) / std_val if std_val > 0 else 0
            
            anomaly_type = AnomalyType.DEMAND_SPIKE if z > 0 else AnomalyType.DEMAND_DROP
            severity = AnomalySeverity.WARNING if scores[idx] > -0.6 else AnomalySeverity.CRITICAL
            
            anomaly = Anomaly(
                product_id=product_id,
                timestamp=data.index[idx] if isinstance(data.index[idx], datetime) else datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                value=float(value),
                expected_value=float(mean_val),
                deviation=float(value - mean_val),
                z_score=float(z),
                message=f"Isolation Forest anomaly (score={scores[idx]:.3f})",
                detection_method="isolation_forest",
                metadata={'anomaly_score': float(scores[idx])}
            )
            anomalies.append(anomaly)
        
        return anomalies


class BusinessRuleDetector:
    """
    Business rule-based anomaly detection.
    
    Detects operational issues like stockout risk and excess inventory.
    """
    
    def __init__(
        self,
        stockout_threshold_days: float = 3.0,
        critical_stockout_days: float = 1.5,
        excess_inventory_days: float = 45.0,
        demand_spike_multiplier: float = 2.0
    ):
        """
        Initialize business rule detector.
        
        Args:
            stockout_threshold_days: Warning threshold for days of stock
            critical_stockout_days: Critical threshold for days of stock
            excess_inventory_days: Days of supply for excess warning
            demand_spike_multiplier: Multiplier for demand spike detection
        """
        self.stockout_threshold_days = stockout_threshold_days
        self.critical_stockout_days = critical_stockout_days
        self.excess_inventory_days = excess_inventory_days
        self.demand_spike_multiplier = demand_spike_multiplier
    
    def detect(
        self,
        product_id: str,
        current_inventory: float,
        daily_demand: float,
        baseline_demand: float,
        safety_stock: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> List[Anomaly]:
        """
        Detect operational anomalies using business rules.
        
        Args:
            product_id: Product identifier
            current_inventory: Current inventory level
            daily_demand: Current daily demand rate
            baseline_demand: Historical baseline demand
            safety_stock: Safety stock level (optional)
            timestamp: Detection timestamp
        
        Returns:
            List of detected anomalies
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if safety_stock is None:
            safety_stock = baseline_demand * 0.5
        
        anomalies = []
        
        # Calculate days of stock
        days_of_stock = current_inventory / daily_demand if daily_demand > 0 else float('inf')
        
        # Stockout risk detection
        if days_of_stock < self.critical_stockout_days:
            anomalies.append(Anomaly(
                product_id=product_id,
                timestamp=timestamp,
                anomaly_type=AnomalyType.STOCKOUT_RISK,
                severity=AnomalySeverity.CRITICAL,
                value=days_of_stock,
                expected_value=self.stockout_threshold_days,
                deviation=days_of_stock - self.stockout_threshold_days,
                z_score=-3.0,
                message=f"CRITICAL: Only {days_of_stock:.1f} days of stock remaining!",
                detection_method="business_rule",
                metadata={
                    'inventory': current_inventory,
                    'daily_demand': daily_demand,
                    'safety_stock': safety_stock
                }
            ))
        elif days_of_stock < self.stockout_threshold_days:
            anomalies.append(Anomaly(
                product_id=product_id,
                timestamp=timestamp,
                anomaly_type=AnomalyType.STOCKOUT_RISK,
                severity=AnomalySeverity.WARNING,
                value=days_of_stock,
                expected_value=self.stockout_threshold_days,
                deviation=days_of_stock - self.stockout_threshold_days,
                z_score=-2.0,
                message=f"WARNING: {days_of_stock:.1f} days of stock remaining",
                detection_method="business_rule",
                metadata={
                    'inventory': current_inventory,
                    'daily_demand': daily_demand
                }
            ))
        
        # Excess inventory detection
        if days_of_stock > self.excess_inventory_days:
            anomalies.append(Anomaly(
                product_id=product_id,
                timestamp=timestamp,
                anomaly_type=AnomalyType.EXCESS_INVENTORY,
                severity=AnomalySeverity.INFO,
                value=days_of_stock,
                expected_value=30.0,  # Target days of supply
                deviation=days_of_stock - 30.0,
                z_score=2.0,
                message=f"Excess inventory: {days_of_stock:.0f} days of supply (target: 30)",
                detection_method="business_rule",
                metadata={'inventory': current_inventory}
            ))
        
        # Demand spike detection
        if daily_demand > baseline_demand * self.demand_spike_multiplier:
            spike_pct = (daily_demand / baseline_demand - 1) * 100
            anomalies.append(Anomaly(
                product_id=product_id,
                timestamp=timestamp,
                anomaly_type=AnomalyType.DEMAND_SPIKE,
                severity=AnomalySeverity.WARNING if spike_pct < 150 else AnomalySeverity.CRITICAL,
                value=daily_demand,
                expected_value=baseline_demand,
                deviation=daily_demand - baseline_demand,
                z_score=spike_pct / 50,  # Approximate z-score
                message=f"Demand spike: {spike_pct:.0f}% above baseline",
                detection_method="business_rule",
                metadata={'spike_pct': spike_pct}
            ))
        
        return anomalies


class AnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.
    
    Combines statistical, ML, and business rule detection for
    comprehensive anomaly identification.
    
    Example:
        >>> detector = AnomalyDetector(methods=['z_score', 'isolation_forest', 'business_rules'])
        >>> detector.fit(historical_data)
        >>> anomalies = detector.detect(current_data)
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        sensitivity: str = 'medium',
        voting_threshold: int = 1
    ):
        """
        Initialize ensemble detector.
        
        Args:
            methods: List of methods to use. Options:
                     'z_score', 'iqr', 'isolation_forest', 'business_rules'
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            voting_threshold: Minimum methods to agree for anomaly
        """
        if methods is None:
            methods = ['z_score', 'business_rules']
        
        self.methods = methods
        self.sensitivity = sensitivity
        self.voting_threshold = voting_threshold
        
        # Configure based on sensitivity
        sensitivity_config = {
            'low': {'z_threshold': 3.5, 'contamination': 0.02},
            'medium': {'z_threshold': 3.0, 'contamination': 0.05},
            'high': {'z_threshold': 2.5, 'contamination': 0.10}
        }
        config = sensitivity_config.get(sensitivity, sensitivity_config['medium'])
        
        # Initialize detectors
        self.detectors = {}
        
        if 'z_score' in methods:
            self.detectors['z_score'] = ZScoreDetector(
                threshold=config['z_threshold']
            )
        
        if 'iqr' in methods:
            self.detectors['iqr'] = IQRDetector(multiplier=1.5)
        
        if 'isolation_forest' in methods and SKLEARN_AVAILABLE:
            self.detectors['isolation_forest'] = IsolationForestDetector(
                contamination=config['contamination']
            )
        
        if 'business_rules' in methods:
            self.detectors['business_rules'] = BusinessRuleDetector()
        
        logger.info(
            "Initialized AnomalyDetector with methods: %s, sensitivity: %s",
            list(self.detectors.keys()), sensitivity
        )
    
    def detect_from_series(
        self,
        data: pd.Series,
        product_id: str = "unknown"
    ) -> List[Anomaly]:
        """
        Detect anomalies from time series data.
        
        Args:
            data: Time series of values
            product_id: Product identifier
        
        Returns:
            List of unique anomalies
        """
        all_anomalies = []
        
        if 'z_score' in self.detectors:
            all_anomalies.extend(
                self.detectors['z_score'].detect(data, product_id)
            )
        
        if 'iqr' in self.detectors:
            all_anomalies.extend(
                self.detectors['iqr'].detect(data, product_id)
            )
        
        # Deduplicate by timestamp
        seen = set()
        unique_anomalies = []
        for a in all_anomalies:
            key = (a.product_id, str(a.timestamp), a.anomaly_type.value)
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(a)
        
        return unique_anomalies
    
    def detect_operational(
        self,
        product_id: str,
        current_inventory: float,
        daily_demand: float,
        baseline_demand: float,
        **kwargs
    ) -> List[Anomaly]:
        """
        Detect operational anomalies using business rules.
        
        Args:
            product_id: Product identifier
            current_inventory: Current inventory level
            daily_demand: Current daily demand
            baseline_demand: Historical baseline
            **kwargs: Additional parameters
        
        Returns:
            List of anomalies
        """
        if 'business_rules' not in self.detectors:
            return []
        
        return self.detectors['business_rules'].detect(
            product_id=product_id,
            current_inventory=current_inventory,
            daily_demand=daily_demand,
            baseline_demand=baseline_demand,
            **kwargs
        )
    
    def get_summary(self, anomalies: List[Anomaly]) -> Dict:
        """
        Get summary statistics for detected anomalies.
        
        Args:
            anomalies: List of anomalies
        
        Returns:
            Summary dictionary
        """
        if not anomalies:
            return {
                'total': 0,
                'by_severity': {},
                'by_type': {},
                'by_method': {}
            }
        
        df = pd.DataFrame([a.to_dict() for a in anomalies])
        
        return {
            'total': len(anomalies),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['anomaly_type'].value_counts().to_dict(),
            'by_method': df['detection_method'].value_counts().to_dict(),
            'critical_count': len(df[df['severity'] == 'critical']),
            'warning_count': len(df[df['severity'] == 'warning'])
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=168, freq='H')
    values = 100 + 20 * np.sin(np.arange(168) * 2 * np.pi / 24) + np.random.normal(0, 5, 168)
    
    # Add anomalies
    values[50] = 300  # Spike
    values[100] = 20  # Drop
    
    data = pd.Series(values, index=dates)
    
    # Test Z-score detector
    print("Z-Score Detection:")
    z_detector = ZScoreDetector(threshold=3.0)
    anomalies = z_detector.detect(data, "TEST_PROD")
    for a in anomalies:
        print(f"  {a.timestamp}: {a.message}")
    
    # Test business rules
    print("\nBusiness Rule Detection:")
    br_detector = BusinessRuleDetector()
    anomalies = br_detector.detect(
        product_id="TEST_PROD",
        current_inventory=50,
        daily_demand=100,
        baseline_demand=80
    )
    for a in anomalies:
        print(f"  [{a.severity.value.upper()}] {a.message}")
    
    # Test ensemble
    print("\nEnsemble Detection:")
    detector = AnomalyDetector(methods=['z_score', 'business_rules'])
    all_anomalies = detector.detect_from_series(data, "TEST_PROD")
    print(f"  Found {len(all_anomalies)} anomalies")
    print(f"  Summary: {detector.get_summary(all_anomalies)}")
