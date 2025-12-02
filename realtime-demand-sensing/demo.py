"""
Real-Time Demand Sensing & Intelligent Replenishment Demo

This demo showcases the complete real-time sensing system:
1. Stream simulation with realistic patterns
2. Real-time demand sensing with drift detection
3. Multi-method anomaly detection (Z-score, IQR, Isolation Forest, Business Rules)
4. Short-term forecasting with ensemble methods
5. Automated replenishment with approval workflows

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
try:
    from src.utils import StreamSimulator
    from src.sensing import DemandSensor
    from src.detection import AnomalyDetector, AlertManager, AnomalySeverity, AnomalyType
    from src.forecasting import ShortTermForecaster, EnsembleForecaster
    from src.replenishment import ReplenishmentEngine, InventoryPosition
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False


def generate_sample_data(days=60, n_products=1):
    """Generate sample sales data with patterns and anomalies."""
    print(f"Generating {days} days of sample data...")
    
    if MODULES_AVAILABLE:
        # Use StreamSimulator
        simulator = StreamSimulator(
            n_products=n_products,
            base_demand_range=(80, 120),
            noise_level=0.1,
            anomaly_probability=0.02,
            seed=42
        )
        df = simulator.generate_historical(days=days)
        
        # Filter to one product for simpler demo
        df = df[df['product_id'] == df['product_id'].iloc[0]].copy()
        
        # Add inventory simulation
        initial_inventory = 5000
        df['inventory'] = initial_inventory - df['sales'].cumsum() * 0.5
        df['inventory'] = np.maximum(100, df['inventory'])
    else:
        # Fallback generation
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        base_demand = 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))
        hour_of_day = dates.hour
        hourly_pattern = 1 + 0.3 * np.sin((hour_of_day - 14) * np.pi / 12)
        demand = base_demand * hourly_pattern + np.random.normal(0, 5, len(dates))
        
        anomaly_indices = np.random.choice(len(dates), size=10, replace=False)
        demand[anomaly_indices] *= np.random.choice([2.5, 0.3], size=10)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'product_id': 'PROD_001',
            'sales': np.maximum(0, demand),
            'inventory': 5000 - np.cumsum(demand) + np.random.normal(0, 100, len(dates))
        })
        df['inventory'] = np.maximum(0, df['inventory'])
    
    return df


def demo_demand_sensing():
    """Demonstrate real-time demand sensing."""
    print("\n" + "="*70)
    print("DEMO 1: REAL-TIME DEMAND SENSING")
    print("="*70 + "\n")
    
    # Generate data
    data = generate_sample_data(days=30)
    
    print(f"Data Summary:")
    print(f"  - Time period: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  - Data points: {len(data)} hourly observations")
    print(f"  - Average hourly sales: {data['sales'].mean():.1f} units")
    print(f"  - Total sales: {data['sales'].sum():.0f} units")
    
    if MODULES_AVAILABLE:
        # Use DemandSensor
        sensor = DemandSensor(alpha=0.3, drift_threshold=2.0, lookback_hours=168)
        
        # Initialize baseline with historical data
        product_id = data['product_id'].iloc[0]
        sensor.initialize_baseline(data[['timestamp', 'sales', 'product_id']])
        
        # Process recent observations
        recent_data = data.tail(24)
        for _, row in recent_data.iterrows():
            result = sensor.update(product_id=product_id, sales=row['sales'], timestamp=row['timestamp'])
        
        # Get current state using get_status_summary
        status = sensor.get_status_summary()
        if not status.empty:
            current_state = status.iloc[0]
            print(f"\nDemand Sensor State:")
            print(f"  - Current level: {current_state['current_demand']:.1f} units/hour")
            print(f"  - Baseline: {current_state['baseline_demand']:.1f} units/hour")
            print(f"  - Z-score: {current_state['z_score']:.2f}")
            print(f"  - Drift detected: {'Yes' if current_state['status'] == 'DRIFT' else 'No'}")
    else:
        # Fallback
        data['rolling_mean'] = data['sales'].rolling(window=24).mean()
        data['rolling_std'] = data['sales'].rolling(window=24).std()
        
        recent_sales = data.tail(24)['sales'].mean()
        baseline = data['rolling_mean'].iloc[-24:].mean()
        change_pct = ((recent_sales - baseline) / baseline) * 100
        
        print(f"\nCurrent Demand Signal:")
        print(f"  - Last 24h average: {recent_sales:.1f} units/hour")
        print(f"  - Baseline (24h MA): {baseline:.1f} units/hour")
        print(f"  - Change: {change_pct:+.1f}%")
    
    # Inventory status
    current_inv = data['inventory'].iloc[-1]
    recent_sales = data.tail(24)['sales'].mean()
    daily_sales = recent_sales * 24
    days_of_stock = current_inv / daily_sales if daily_sales > 0 else 999
    
    print(f"\nInventory Status:")
    print(f"  - Current inventory: {current_inv:.0f} units")
    print(f"  - Daily sales rate: {daily_sales:.0f} units/day")
    print(f"  - Days of stock: {days_of_stock:.1f} days")
    
    if days_of_stock < 3:
        print(f"  - 🚨 CRITICAL: Stockout risk!")
    elif days_of_stock < 7:
        print(f"  - ⚠️  WARNING: Low inventory")
    else:
        print(f"  - ✓ Inventory healthy")
    
    return data


def demo_anomaly_detection():
    """Demonstrate anomaly detection."""
    print("\n\n" + "="*70)
    print("DEMO 2: MULTI-METHOD ANOMALY DETECTION")
    print("="*70 + "\n")
    
    data = generate_sample_data(days=30)
    
    if MODULES_AVAILABLE:
        # Use AnomalyDetector ensemble
        detector = AnomalyDetector(
            methods=['z_score', 'iqr', 'business_rules'],
            sensitivity='medium',
            voting_threshold=1
        )
        
        # Detect anomalies from sales series
        product_id = data['product_id'].iloc[0]
        anomalies = detector.detect_from_series(data['sales'], product_id=product_id)
        
        print(f"Anomaly Detection Results (Ensemble Method):")
        print(f"  - Methods: Z-Score, IQR, Business Rules")
        print(f"  - Total anomalies detected: {len(anomalies)}")
        
        # Count by severity
        critical = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        warning = sum(1 for a in anomalies if a.severity == AnomalySeverity.WARNING)
        info = sum(1 for a in anomalies if a.severity == AnomalySeverity.INFO)
        
        print(f"  - Critical: {critical}, Warning: {warning}, Info: {info}")
        
        if anomalies:
            print(f"\nRecent Anomalies:")
            for anomaly in anomalies[-5:]:
                severity_icon = "🔴" if anomaly.severity == AnomalySeverity.CRITICAL else "🟡" if anomaly.severity == AnomalySeverity.WARNING else "🔵"
                print(f"  {severity_icon} [{anomaly.anomaly_type.value}] Value: {anomaly.value:.1f}")
                print(f"     {anomaly.message}")
        
        # Get summary
        summary = detector.get_summary(anomalies)
        
        print(f"\nDetector Summary:")
        print(f"  - Total anomalies: {summary['total']}")
        print(f"  - By severity: {summary['by_severity']}")
    else:
        # Fallback to simple z-score
        data['zscore'] = (data['sales'] - data['sales'].mean()) / data['sales'].std()
        anomalies = data[np.abs(data['zscore']) > 3]
        
        print(f"Anomaly Detection Results:")
        print(f"  - Method: Z-score (threshold = 3.0)")
        print(f"  - Anomalies detected: {len(anomalies)}")
        print(f"  - Anomaly rate: {len(anomalies)/len(data)*100:.2f}%")
        
        if len(anomalies) > 0:
            print(f"\nRecent Anomalies:")
            for idx, row in anomalies.tail(5).iterrows():
                anomaly_type = "SPIKE" if row['zscore'] > 0 else "DROP"
                print(f"  - {row['timestamp']}: {anomaly_type} - Sales: {row['sales']:.0f} units (z={row['zscore']:.2f})")
    
    return anomalies


def demo_forecasting():
    """Demonstrate short-term forecasting."""
    print("\n\n" + "="*70)
    print("DEMO 3: SHORT-TERM ENSEMBLE FORECASTING")
    print("="*70 + "\n")
    
    data = generate_sample_data(days=60)
    
    if MODULES_AVAILABLE:
        # Prepare data
        forecast_data = data[['timestamp', 'sales']].copy()
        forecast_data.columns = ['timestamp', 'value']
        
        # Use EnsembleForecaster
        forecaster = EnsembleForecaster(
            models=['ewm', 'moving_average'],
            confidence=0.95
        )
        forecaster.fit(forecast_data)
        
        # Generate forecast
        horizon = 48  # 48 hours
        forecasts = forecaster.predict(horizon=horizon)
        
        print(f"Forecasting Configuration:")
        print(f"  - Method: Ensemble (EWM + Moving Average)")
        print(f"  - Horizon: {horizon} hours")
        print(f"  - Confidence: 95%")
        print(f"  - Historical data: {len(data)} hours")
        
        print(f"\nForecast Summary:")
        if forecasts:
            total_24h = sum(f.value for f in forecasts[:min(24, len(forecasts))])
            total_48h = sum(f.value for f in forecasts)
            
            print(f"  - Next 24h forecast: {total_24h:.0f} units")
            print(f"  - Next 48h forecast: {total_48h:.0f} units")
            
            print(f"\nHourly Forecasts (Next 12 hours):")
            print(f"  {'Hour':<8} {'Forecast':<12} {'Lower':<12} {'Upper'}")
            print(f"  {'-'*50}")
            for i, f in enumerate(forecasts[:min(12, len(forecasts))]):
                print(f"  H+{i+1:<5} {f.value:>8.1f}     {f.lower_bound:>8.1f}     {f.upper_bound:>8.1f}")
        else:
            print(f"  No forecasts generated")
        
        # Use ShortTermForecaster for comparison
        st_forecaster = ShortTermForecaster(model='ensemble')
        st_forecaster.fit(forecast_data)
        st_result = st_forecaster.forecast(horizon=24)
        print(f"\nShortTermForecaster comparison (24h): {st_result['forecast'].sum():.0f} units")
        
    else:
        # Fallback to simple exponential smoothing
        alpha = 0.3
        data['forecast'] = data['sales'].ewm(alpha=alpha, adjust=False).mean()
        last_value = data['forecast'].iloc[-1]
        trend = data['forecast'].diff().tail(24).mean()
        
        horizon = 7
        forecast_values = [last_value + trend * i for i in range(1, horizon + 1)]
        
        print(f"Forecasting Configuration:")
        print(f"  - Method: Exponential Smoothing (α={alpha})")
        print(f"  - Horizon: {horizon} days")
        print(f"  - Historical data: {len(data)} hours")
        
        print(f"\n7-Day Forecast (Daily Averages):")
        for day in range(horizon):
            forecast_value = forecast_values[day] * 24
            trend_indicator = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            print(f"  Day {day+1}: {forecast_value:>8.0f} units {trend_indicator}")
        
        total_forecast = sum(forecast_values) * 24
        print(f"\n  Total 7-day forecast: {total_forecast:.0f} units")
    
    # Forecast accuracy on historical data (both paths)
    last_week = data.tail(7*24).copy()
    last_week['forecast'] = last_week['sales'].ewm(alpha=0.3, adjust=False).mean()
    mae = np.abs(last_week['sales'] - last_week['forecast']).mean()
    mape = (np.abs((last_week['sales'] - last_week['forecast']) / last_week['sales'].replace(0, 1)).mean()) * 100
    
    print(f"\nForecast Accuracy (Last 7 Days):")
    print(f"  - MAE: {mae:.1f} units")
    print(f"  - MAPE: {mape:.1f}%")
    
    return forecasts if MODULES_AVAILABLE else forecast_values


def demo_replenishment_engine():
    """Demonstrate automated replenishment triggers."""
    print("\n\n" + "="*70)
    print("DEMO 4: AUTOMATED REPLENISHMENT ENGINE")
    print("="*70 + "\n")
    
    # Product portfolio
    products = pd.DataFrame({
        'product_id': [f'PROD_{i:03d}' for i in range(1, 11)],
        'on_hand': [500, 1200, 300, 800, 2000, 150, 600, 900, 400, 1100],
        'on_order': [0, 200, 0, 0, 300, 100, 0, 0, 50, 0],
        'allocated': [50, 100, 30, 80, 200, 20, 60, 90, 40, 100],
        'safety_stock': [200, 300, 250, 180, 400, 160, 220, 190, 260, 170],
        'reorder_point': [500, 750, 730, 450, 800, 560, 550, 570, 650, 510],
        'avg_daily_demand': [100, 150, 120, 90, 200, 80, 110, 95, 130, 85],
        'lead_time_days': [3, 3, 4, 3, 2, 5, 3, 4, 3, 4],
        'unit_cost': [25, 40, 35, 28, 50, 22, 30, 38, 32, 26]
    })
    
    if MODULES_AVAILABLE:
        # Use ReplenishmentEngine
        engine = ReplenishmentEngine(
            target_dos=14,
            auto_approve_limit=1000,
            auto_approve_emergency=True
        )
        
        # Update positions
        engine.update_positions(products)
        
        # Run replenishment cycle
        results = engine.run_cycle()
        
        print(f"Replenishment Engine Results:")
        print(f"  - Positions evaluated: {len(products)}")
        print(f"  - Triggers fired: {results['triggers']}")
        print(f"  - Orders generated: {results['orders_generated']}")
        print(f"  - Auto-approved: {results['auto_approved']}")
        print(f"  - Pending approval: {results['pending_approval']}")
        print(f"  - Total order value: ${results['total_value']:,.2f}")
        
        if results['orders']:
            print(f"\nGenerated Orders:")
            print(f"  {'-'*90}")
            print(f"  {'Order ID':<20} {'Product':<12} {'Qty':<10} {'Priority':<12} {'Status':<18} {'Value'}")
            print(f"  {'-'*90}")
            
            for order in results['orders']:
                priority_icon = "🚨" if order.priority.name == 'EMERGENCY' else "⚠️" if order.priority.name == 'HIGH' else "📦"
                print(f"  {priority_icon} {order.id:<18} {order.product_id:<12} {order.quantity:>6.0f}    "
                      f"{order.priority.name:<12} {order.status.value:<18} ${order.total_cost:>10,.2f}")
        
        summary = engine.get_summary()
        print(f"\nEngine Summary:")
        print(f"  - Positions below ROP: {summary['positions_below_rop']}")
        print(f"  - Emergency orders: {summary['emergency_orders']}")
        print(f"  - Pending order value: ${summary['pending_value']:,.2f}")
        
    else:
        # Fallback
        products['available'] = products['on_hand'] + products['on_order'] - products['allocated']
        products['days_of_stock'] = products['available'] / products['avg_daily_demand']
        
        products['trigger_type'] = 'None'
        products.loc[products['available'] <= products['reorder_point'], 'trigger_type'] = 'Reorder Point'
        products.loc[products['days_of_stock'] < 3, 'trigger_type'] = 'Stockout Risk'
        products.loc[products['days_of_stock'] < 1.5, 'trigger_type'] = 'Critical'
        
        products['order_qty'] = 0
        needs_repl = products[products['trigger_type'] != 'None']
        
        for idx, row in needs_repl.iterrows():
            target_inventory = (row['avg_daily_demand'] * 14) + row['safety_stock']
            order_qty = max(0, target_inventory - row['available'])
            products.at[idx, 'order_qty'] = order_qty
        
        products['order_value'] = products['order_qty'] * products['unit_cost']
        
        print(f"Portfolio Summary:")
        print(f"  - Total products: {len(products)}")
        print(f"  - Products needing replenishment: {len(needs_repl)}")
        print(f"  - Critical alerts: {len(products[products['trigger_type'] == 'Critical'])}")
        print(f"  - Total order value: ${products['order_value'].sum():,.2f}")
        
        print(f"\nReplenishment Queue:")
        for _, row in products[products['trigger_type'] != 'None'].iterrows():
            priority_icon = "🚨" if row['trigger_type'] == 'Critical' else "⚠️"
            print(f"  {priority_icon} {row['product_id']}: Order {row['order_qty']:.0f} units (${row['order_value']:,.2f})")
    
    return products


def main():
    """Run all demos."""
    print("\n")
    print("="*70)
    print("REAL-TIME DEMAND SENSING - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo showcases real-time demand sensing, anomaly detection,")
    print("forecasting, and automated replenishment for operational excellence.")
    
    if MODULES_AVAILABLE:
        print("\n✅ All modules loaded successfully!")
    else:
        print("\n⚠️  Some modules not available. Using fallback implementations.")
    
    try:
        # Demo 1: Demand Sensing
        data = demo_demand_sensing()
        
        # Demo 2: Anomaly Detection
        anomalies = demo_anomaly_detection()
        
        # Demo 3: Forecasting
        forecast = demo_forecasting()
        
        # Demo 4: Replenishment
        replenishment = demo_replenishment_engine()
        
        print("\n\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\n📊 Key Capabilities Demonstrated:")
        print("  1. StreamSimulator - Realistic demand pattern generation")
        print("  2. DemandSensor - Real-time demand estimation with drift detection")
        print("  3. AnomalyDetector - Multi-method anomaly detection ensemble")
        print("  4. EnsembleForecaster - Short-term demand forecasting")
        print("  5. ReplenishmentEngine - Automated ordering with approvals")
        
        print("\n💰 Expected Business Impact:")
        print("  - 25-30% reduction in stockouts")
        print("  - 80%+ automation of routine decisions")
        print("  - 2-3 day early warning on demand shifts")
        print("  - Improved forecast accuracy (MAPE < 15%)")
        
        print("\n🚀 Next Steps:")
        print("  - Launch interactive dashboard: streamlit run app.py")
        print("  - See notebooks/ for detailed analysis")
        print("  - Review docs/ for methodology")
        print("  - Run tests: pytest tests/")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nNote: This demo requires the following packages:")
        print("  pip install pandas numpy scikit-learn streamlit scipy")
        print("\nRun: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
