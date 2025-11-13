# CLAUDE.md - Project 005: Real-Time Demand Sensing & Intelligent Replenishment

This file provides guidance to Claude Code when working with the **Real-Time Demand Sensing** project.

## Project Overview

An intelligent system that senses demand signals in real-time, detects anomalies, and triggers automated replenishment decisions with an interactive dashboard for monitoring and what-if analysis. This project transforms traditional batch forecasting into a **real-time operational system** combining streaming data simulation, anomaly detection, and automated decision-making.

**Status**: 🚧 Template Ready (Architecture defined, awaiting implementation)
**Implementation Priority**: Medium-High
**Complexity**: Advanced (Real-time systems, dashboards, automation)

## Quick Start

```bash
# Navigate to project
cd data-science-portfolio/project-005-realtime-demand-sensing

# Activate shared virtual environment
source ../activate.sh

# Install dependencies (when implementing)
pip install -r requirements.txt
pip install -e .

# Run demo (when implemented)
python demo.py

# Launch interactive dashboard (when implemented)
streamlit run app.py
```

## Project Architecture

### Directory Structure

```
project-005-realtime-demand-sensing/
├── src/
│   ├── sensing/
│   │   ├── signal_processor.py    # Real-time signal processing
│   │   ├── demand_sensor.py       # Demand estimation
│   │   └── external_signals.py    # Weather, events integration
│   ├── detection/
│   │   ├── anomaly_detector.py    # Anomaly detection
│   │   ├── threshold_monitor.py   # KPI monitoring
│   │   └── alert_manager.py       # Alert system
│   ├── forecasting/
│   │   ├── short_term.py          # Hourly/daily forecasts
│   │   ├── prophet_model.py       # Prophet for seasonality
│   │   └── ensemble.py            # Model ensemble
│   ├── replenishment/
│   │   ├── trigger_engine.py      # Replenishment triggers
│   │   ├── order_generator.py     # PO generation
│   │   └── priority_ranker.py     # Urgency ranking
│   ├── dashboard/
│   │   ├── app.py                 # Main Streamlit app
│   │   ├── components/            # Dashboard components
│   │   │   ├── monitoring.py
│   │   │   ├── alerts.py
│   │   │   ├── replenishment.py
│   │   │   └── whatif.py
│   │   └── callbacks.py           # Interactive callbacks
│   └── utils/
│       ├── stream_simulator.py    # Data streaming simulation
│       ├── cache_manager.py       # State management
│       └── notification.py        # Alert notifications
├── notebooks/
│   ├── 01_demand_signal_analysis.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_short_term_forecasting.ipynb
│   └── 04_replenishment_rules.ipynb
├── data/
│   ├── real_time/                 # Streaming data buffer
│   └── historical/                # Historical baselines
├── models/                        # Trained models
├── config/
│   └── config.yaml
├── tests/
├── app.py                         # Main Streamlit app
├── demo.py
└── README.md
```

### Core Concepts

#### 1. Real-Time Demand Sensing

**Definition**: Continuously monitor and estimate current demand from multiple signals.

**Key Differences from Batch Forecasting**:

| Aspect | Batch Forecasting | Real-Time Sensing |
|--------|-------------------|-------------------|
| **Frequency** | Daily/Weekly | Hourly/Continuous |
| **Latency** | Hours to days | Minutes |
| **Data Sources** | Historical sales | Sales + web + POS + inventory |
| **Decision Trigger** | Scheduled | Event-driven |
| **Human Involvement** | High (review forecasts) | Low (automated actions) |

#### 2. Anomaly Detection

**Purpose**: Identify unusual patterns that require attention.

**Anomaly Types**:
1. **Demand Spike**: Sudden increase (viral product, competitor stockout)
2. **Demand Drop**: Sudden decrease (quality issue, competitor promotion)
3. **Stockout Risk**: Inventory falling below safety threshold
4. **Excess Inventory**: Days of supply exceeding target
5. **Pattern Break**: Deviation from seasonal baseline

#### 3. Automated Replenishment

**Trigger-Based System**: Execute orders when conditions met.

**Example Rules**:
```python
if inventory < reorder_point:
    trigger_order(quantity=eoq)

if demand_spike_detected() and inventory < 2 * daily_demand:
    trigger_expedited_order()

if days_of_supply > 60:
    trigger_markdown()
```

### Key Modules

#### 1. Demand Sensor (`src/sensing/demand_sensor.py`)

**Purpose**: Estimate current demand from real-time signals.

**Key Classes**:

**`DemandSensor`** - Real-time demand estimator
- `get_current_demand(product_id, signals)` - Latest demand estimate
- `update_baseline(historical_data)` - Refresh baseline
- `detect_drift(current, baseline)` - Drift detection

**Signal Fusion Methods**:

1. **Weighted Average**:
```python
def estimate_demand(self, signals):
    """Combine multiple signals with weights."""
    weights = {
        'sales': 0.50,
        'web_traffic': 0.20,
        'pos_scans': 0.15,
        'inventory_velocity': 0.15
    }

    demand_estimate = sum(
        weights[source] * signals[source]
        for source in weights
    )

    return demand_estimate
```

2. **Exponential Smoothing with Drift Detection**:
```python
class DemandSensor:
    def __init__(self, alpha=0.3, drift_threshold=2.0):
        self.alpha = alpha  # Smoothing parameter
        self.drift_threshold = drift_threshold
        self.baseline = None

    def update(self, new_observation):
        """Update demand estimate with new data."""
        if self.baseline is None:
            self.baseline = new_observation
        else:
            # Exponential smoothing
            self.baseline = self.alpha * new_observation + \
                           (1 - self.alpha) * self.baseline

        return self.baseline

    def detect_drift(self, observation):
        """Detect if current demand has drifted from baseline."""
        if self.baseline is None:
            return False

        z_score = (observation - self.baseline) / self.std
        return abs(z_score) > self.drift_threshold
```

3. **Kalman Filter** (advanced):
```python
from filterpy.kalman import KalmanFilter

def kalman_demand_sensing(observations, process_variance=0.01,
                          measurement_variance=0.1):
    """Use Kalman filter for demand sensing."""
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State: [demand, demand_trend]
    kf.x = np.array([observations[0], 0])

    # State transition matrix
    kf.F = np.array([[1, 1],    # demand + trend
                     [0, 1]])    # trend remains

    # Measurement matrix
    kf.H = np.array([[1, 0]])   # Observe demand only

    # Covariance matrices
    kf.P *= 1000
    kf.R = measurement_variance
    kf.Q = process_variance * np.eye(2)

    estimates = []
    for obs in observations:
        kf.predict()
        kf.update(obs)
        estimates.append(kf.x[0])

    return estimates
```

**Usage**:
```python
from src.sensing import DemandSensor

sensor = DemandSensor(
    lookback_hours=24,
    update_frequency='1H'
)

# Initialize with historical baseline
sensor.update_baseline(historical_sales)

# Get current demand estimate
current_demand = sensor.get_current_demand(
    product_id='FOODS_1_001',
    include_signals=['sales', 'web_traffic', 'pos_scans']
)

print(f"Current Demand: {current_demand:.1f} units/day")
print(f"Baseline: {sensor.baseline:.1f} units/day")

if sensor.detect_drift(current_demand):
    print("⚠️ Demand drift detected!")
```

#### 2. Anomaly Detector (`src/detection/anomaly_detector.py`)

**Purpose**: Identify unusual patterns requiring attention.

**Key Classes**:

**`AnomalyDetector`** - Multi-method anomaly detection
- `detect_anomalies(data, method)` - Run detection
- `classify_severity(anomaly)` - Severity scoring
- `generate_alert(anomaly)` - Create actionable alert

**Detection Methods**:

1. **Statistical Methods**:
```python
def z_score_detection(data, threshold=3.0):
    """Detect outliers using z-score."""
    mean = data.mean()
    std = data.std()
    z_scores = np.abs((data - mean) / std)
    anomalies = z_scores > threshold
    return anomalies

def iqr_detection(data, multiplier=1.5):
    """Detect outliers using IQR."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    anomalies = (data < lower_bound) | (data > upper_bound)
    return anomalies
```

2. **Machine Learning Methods**:
```python
from sklearn.ensemble import IsolationForest

def isolation_forest_detection(data, contamination=0.1):
    """Use Isolation Forest for anomaly detection."""
    clf = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    # Reshape for sklearn
    X = data.values.reshape(-1, 1)

    # Fit and predict (-1 = anomaly, 1 = normal)
    predictions = clf.fit_predict(X)
    anomalies = predictions == -1

    # Get anomaly scores
    scores = clf.score_samples(X)

    return anomalies, scores
```

3. **LSTM Autoencoder** (deep learning):
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector

def lstm_autoencoder_detection(data, sequence_length=24, threshold=0.95):
    """Use LSTM autoencoder for time series anomaly detection."""

    # Build autoencoder
    model = Sequential([
        LSTM(128, input_shape=(sequence_length, 1)),
        RepeatVector(sequence_length),
        LSTM(128, return_sequences=True),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Prepare sequences
    X = create_sequences(data, sequence_length)

    # Train on normal data
    model.fit(X, X, epochs=50, batch_size=32, verbose=0)

    # Predict and calculate reconstruction error
    X_pred = model.predict(X)
    mse = np.mean(np.square(X - X_pred), axis=(1, 2))

    # Anomalies = high reconstruction error
    threshold_value = np.percentile(mse, threshold * 100)
    anomalies = mse > threshold_value

    return anomalies, mse
```

4. **Business Rule-Based Detection**:
```python
class BusinessRuleDetector:
    def __init__(self, rules):
        self.rules = rules

    def detect(self, product_data):
        """Apply business rules for anomaly detection."""
        anomalies = []

        # Rule 1: Stockout risk
        if product_data['inventory'] < product_data['safety_stock']:
            anomalies.append({
                'type': 'stockout_risk',
                'severity': 'critical',
                'days_to_stockout': product_data['inventory'] /
                                   product_data['daily_demand'],
                'message': f"Stockout risk in {days_to_stockout:.1f} days"
            })

        # Rule 2: Demand spike
        if product_data['current_demand'] > 2 * product_data['baseline']:
            anomalies.append({
                'type': 'demand_spike',
                'severity': 'warning',
                'spike_pct': (product_data['current_demand'] /
                             product_data['baseline'] - 1) * 100,
                'message': f"Demand up {spike_pct:.0f}%"
            })

        # Rule 3: Excess inventory
        if product_data['days_of_supply'] > 60:
            anomalies.append({
                'type': 'excess_inventory',
                'severity': 'info',
                'dos': product_data['days_of_supply'],
                'message': f"{dos:.0f} days of supply (target: 30)"
            })

        return anomalies
```

**Usage**:
```python
from src.detection import AnomalyDetector

detector = AnomalyDetector(
    method='ensemble',  # Combine multiple methods
    sensitivity='medium'
)

# Detect anomalies
anomalies = detector.detect(
    product_id='FOODS_1_001',
    current_inventory=15,
    daily_sales=42,
    safety_stock=20,
    baseline_demand=35
)

# Display alerts
for anomaly in anomalies:
    print(f"[{anomaly['severity'].upper()}] {anomaly['type']}: "
          f"{anomaly['message']}")
```

#### 3. Short-Term Forecaster (`src/forecasting/short_term.py`)

**Purpose**: Generate next 1-7 day forecasts for replenishment.

**Key Models**:

1. **Prophet** (Meta's forecasting library):
```python
from prophet import Prophet

class ProphetForecaster:
    def __init__(self, seasonality_mode='multiplicative'):
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True
        )

    def fit(self, df):
        """Fit Prophet model."""
        # Prophet requires 'ds' (date) and 'y' (value) columns
        prophet_df = df.rename(columns={'date': 'ds', 'sales': 'y'})
        self.model.fit(prophet_df)

    def forecast(self, periods=7):
        """Forecast next N days."""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

2. **SARIMAX** (Seasonal ARIMA):
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, data):
        """Fit SARIMAX model."""
        self.model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order
        )
        self.fitted = self.model.fit(disp=False)

    def forecast(self, steps=7):
        """Forecast next N steps."""
        forecast = self.fitted.forecast(steps=steps)
        conf_int = self.fitted.get_forecast(steps=steps).conf_int()

        return {
            'forecast': forecast,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        }
```

3. **Ensemble** (combine multiple models):
```python
class EnsembleForecaster:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def fit(self, data):
        """Fit all models."""
        for model in self.models:
            model.fit(data)

    def forecast(self, steps=7):
        """Weighted ensemble forecast."""
        forecasts = [model.forecast(steps) for model in self.models]

        ensemble_forecast = sum(
            w * f for w, f in zip(self.weights, forecasts)
        ) / sum(self.weights)

        return ensemble_forecast

    def update_weights(self, validation_data):
        """Update weights based on recent performance."""
        errors = []
        for model in self.models:
            predictions = model.predict(validation_data)
            error = mean_absolute_error(validation_data, predictions)
            errors.append(error)

        # Inverse error weighting
        inverse_errors = 1 / np.array(errors)
        self.weights = inverse_errors / inverse_errors.sum()
```

#### 4. Replenishment Trigger Engine (`src/replenishment/trigger_engine.py`)

**Purpose**: Automatically generate replenishment orders based on rules.

**Key Classes**:

**`TriggerEngine`** - Rule-based replenishment automation
- `evaluate_triggers(product_data)` - Check all trigger conditions
- `generate_order(product_id, trigger_reason)` - Create purchase order
- `prioritize_orders(orders)` - Rank by urgency

**Trigger Rules**:

```python
class TriggerEngine:
    def __init__(self, rules):
        self.rules = rules

    def evaluate_triggers(self, product_data):
        """Evaluate all trigger conditions."""
        triggers = []

        # Trigger 1: Inventory below reorder point
        if product_data['inventory'] <= product_data['reorder_point']:
            triggers.append({
                'type': 'reorder_point',
                'priority': 'high',
                'quantity': product_data['eoq'],
                'reason': f"Inventory ({product_data['inventory']}) "
                         f"at reorder point ({product_data['reorder_point']})"
            })

        # Trigger 2: Demand spike detected
        if (product_data['demand_spike_detected'] and
            product_data['inventory'] < 3 * product_data['daily_demand']):
            triggers.append({
                'type': 'demand_spike',
                'priority': 'urgent',
                'quantity': 2 * product_data['eoq'],  # Order more
                'expedited': True,
                'reason': f"Demand spike detected, low inventory"
            })

        # Trigger 3: Forecast excess
        forecast_demand = product_data['forecast_7day']
        available_inventory = product_data['inventory'] + \
                             product_data['in_transit']

        if forecast_demand > available_inventory + product_data['safety_stock']:
            shortfall = forecast_demand - available_inventory
            triggers.append({
                'type': 'forecast_excess',
                'priority': 'medium',
                'quantity': shortfall + product_data['safety_stock'],
                'reason': f"7-day forecast ({forecast_demand:.0f}) "
                         f"exceeds available ({available_inventory:.0f})"
            })

        return triggers

    def generate_orders(self, products_data):
        """Generate orders for all products."""
        all_orders = []

        for product_id, data in products_data.items():
            triggers = self.evaluate_triggers(data)

            for trigger in triggers:
                order = {
                    'product_id': product_id,
                    'type': trigger['type'],
                    'priority': trigger['priority'],
                    'quantity': trigger['quantity'],
                    'expedited': trigger.get('expedited', False),
                    'reason': trigger['reason'],
                    'estimated_cost': trigger['quantity'] * data['unit_cost'],
                    'auto_approve': self.should_auto_approve(trigger, data)
                }
                all_orders.append(order)

        # Prioritize orders
        all_orders = self.prioritize_orders(all_orders)

        return all_orders

    def should_auto_approve(self, trigger, product_data):
        """Determine if order can be auto-approved."""
        # Auto-approve if:
        # 1. Routine reorder (not demand spike)
        # 2. A or B item (high value)
        # 3. Order value < threshold
        # 4. Supplier has capacity

        if trigger['type'] != 'reorder_point':
            return False  # Review non-routine orders

        if product_data['abc_class'] in ['A', 'B']:
            return trigger['quantity'] * product_data['unit_cost'] < 5000

        return True  # Auto-approve low-value C items

    def prioritize_orders(self, orders):
        """Rank orders by urgency."""
        priority_scores = {
            'urgent': 3,
            'high': 2,
            'medium': 1,
            'low': 0
        }

        # Calculate priority score
        for order in orders:
            score = priority_scores[order['priority']]

            # Boost score for high-value items
            if order.get('abc_class') == 'A':
                score += 2
            elif order.get('abc_class') == 'B':
                score += 1

            # Boost for stockout risk
            if order.get('days_to_stockout', float('inf')) < 3:
                score += 3

            order['priority_score'] = score

        # Sort by priority score (descending)
        return sorted(orders, key=lambda x: x['priority_score'], reverse=True)
```

#### 5. Streamlit Dashboard (`src/dashboard/app.py`)

**Purpose**: Interactive real-time monitoring and control interface.

**Dashboard Components**:

```python
import streamlit as st
import plotly.graph_objects as go

# Main app
def main():
    st.set_page_config(
        page_title="Real-Time Demand Sensing",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Real-Time Demand Sensing & Replenishment")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        refresh_interval = st.selectbox(
            "Refresh Interval",
            [5, 10, 30, 60],
            index=1
        )
        st.write(f"Auto-refresh every {refresh_interval}s")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Monitoring",
        "🚨 Alerts",
        "📦 Replenishment",
        "🔮 What-If",
        "📊 KPIs"
    ])

    with tab1:
        monitoring_dashboard()

    with tab2:
        alerts_dashboard()

    with tab3:
        replenishment_dashboard()

    with tab4:
        whatif_dashboard()

    with tab5:
        kpi_dashboard()

def monitoring_dashboard():
    """Real-time demand monitoring."""
    st.header("Real-Time Demand Monitoring")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Demand",
            value="42 units/day",
            delta="+18% vs baseline"
        )

    with col2:
        st.metric(
            label="Inventory Level",
            value="1,250 units",
            delta="-5% (2.5 days supply)"
        )

    with col3:
        st.metric(
            label="Active Alerts",
            value="8",
            delta="3 critical"
        )

    with col4:
        st.metric(
            label="Pending Orders",
            value="12",
            delta="8 auto-approved"
        )

    # Real-time chart
    st.subheader("Demand Trend (Last 24 Hours)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_data,
        y=actual_demand,
        name='Actual Demand',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=time_data,
        y=baseline_demand,
        name='Baseline',
        line=dict(dash='dash')
    ))

    st.plotly_chart(fig, use_container_width=True)

def alerts_dashboard():
    """Anomaly alerts with drill-down."""
    st.header("🚨 Anomaly Alerts")

    # Filter
    severity_filter = st.multiselect(
        "Filter by Severity",
        ["Critical", "Warning", "Info"],
        default=["Critical", "Warning"]
    )

    # Alerts table
    alerts_df = get_active_alerts()
    filtered_alerts = alerts_df[alerts_df['severity'].isin(severity_filter)]

    for _, alert in filtered_alerts.iterrows():
        with st.expander(f"[{alert['severity']}] {alert['type']} - {alert['product_id']}"):
            st.write(f"**Message**: {alert['message']}")
            st.write(f"**Detected**: {alert['timestamp']}")
            st.write(f"**Current Value**: {alert['current_value']}")
            st.write(f"**Expected Value**: {alert['expected_value']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                    acknowledge_alert(alert['id'])

            with col2:
                if st.button("Create Order", key=f"order_{alert['id']}"):
                    create_order_from_alert(alert)

            with col3:
                if st.button("Dismiss", key=f"dismiss_{alert['id']}"):
                    dismiss_alert(alert['id'])

def replenishment_dashboard():
    """Replenishment queue and approval."""
    st.header("📦 Replenishment Queue")

    # Pending orders
    orders_df = get_pending_orders()

    st.subheader(f"Pending Orders ({len(orders_df)})")

    for _, order in orders_df.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

        with col1:
            st.write(f"**{order['product_id']}**")
            st.write(f"{order['reason']}")

        with col2:
            st.write(f"Quantity: {order['quantity']}")
            st.write(f"Cost: ${order['cost']:,.0f}")

        with col3:
            priority_color = {
                'urgent': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }
            st.write(f"{priority_color[order['priority']]} {order['priority']}")

        with col4:
            if order['auto_approved']:
                st.success("Auto-approved")
            else:
                if st.button("Approve", key=f"approve_{order['id']}"):
                    approve_order(order['id'])
                if st.button("Reject", key=f"reject_{order['id']}"):
                    reject_order(order['id'])

def whatif_dashboard():
    """Scenario analysis."""
    st.header("🔮 What-If Analysis")

    product_id = st.selectbox("Select Product", get_product_list())

    st.subheader("Test Scenarios")

    col1, col2 = st.columns(2)

    with col1:
        demand_change = st.slider(
            "Demand Change (%)",
            -50, 100, 0
        )

    with col2:
        lead_time_change = st.slider(
            "Lead Time Change (days)",
            -7, 14, 0
        )

    if st.button("Run Scenario"):
        result = run_scenario(
            product_id,
            demand_change,
            lead_time_change
        )

        st.subheader("Scenario Results")
        st.write(f"**New Reorder Point**: {result['rop']:.0f}")
        st.write(f"**New Safety Stock**: {result['ss']:.0f}")
        st.write(f"**Days to Stockout**: {result['days_to_stockout']:.1f}")

def kpi_dashboard():
    """KPI tracking."""
    st.header("📊 Key Performance Indicators")

    # Time range
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )

    kpis = get_kpis(time_range)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Service Level",
            f"{kpis['service_level']:.1f}%",
            delta=f"+{kpis['service_level_change']:.1f}%"
        )

    with col2:
        st.metric(
            "Avg Inventory",
            f"{kpis['avg_inventory']:,.0f}",
            delta=f"-{kpis['inventory_change']:.0f}"
        )

    with col3:
        st.metric(
            "Stockout Rate",
            f"{kpis['stockout_rate']:.1f}%",
            delta=f"-{kpis['stockout_change']:.1f}%"
        )
```

**Running the Dashboard**:
```bash
streamlit run app.py
```

## Configuration

### config/config.yaml

```yaml
data:
  historical_data_path: "../project-001-demand-forecasting-system/data/processed"
  real_time_buffer: "data/real_time"
  cache_ttl: 300  # seconds

sensing:
  update_frequency: "1H"  # Update every hour
  lookback_hours: 24
  signal_weights:
    sales: 0.50
    web_traffic: 0.20
    pos_scans: 0.15
    inventory_velocity: 0.15

detection:
  anomaly_methods:
    - z_score
    - isolation_forest
    - business_rules
  ensemble_method: "voting"  # voting, weighted_average
  sensitivity: "medium"  # low, medium, high

  thresholds:
    z_score: 3.0
    isolation_contamination: 0.1
    drift_threshold: 2.0

  severity_rules:
    critical:
      - stockout_risk_days < 2
      - demand_spike > 2x baseline
    warning:
      - stockout_risk_days < 5
      - demand_change > 50%
    info:
      - days_of_supply > 60

forecasting:
  methods:
    - prophet
    - sarimax
  ensemble_weights:
    prophet: 0.6
    sarimax: 0.4
  forecast_horizon: 7  # days

replenishment:
  triggers:
    - reorder_point
    - demand_spike
    - forecast_excess
  auto_approve_threshold: 5000  # $
  auto_approve_classes: ["B", "C"]  # A items require review

  priority_rules:
    urgent: stockout_risk_days < 2
    high: stockout_risk_days < 5 or abc_class == "A"
    medium: routine reorder
    low: safety stock replenishment

dashboard:
  refresh_interval: 10  # seconds
  max_alerts_display: 50
  kpi_time_ranges: [7, 30, 90]  # days

notifications:
  email_enabled: true
  sms_enabled: false
  slack_enabled: true

  alert_levels:
    critical:
      - email
      - sms
      - slack
    warning:
      - email
      - slack
    info:
      - slack
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure
- [ ] Implement data streaming simulator
- [ ] Create baseline demand calculations
- [ ] Build cache management system
- [ ] Write basic tests

### Phase 2: Demand Sensing (Week 3-4)
- [ ] Implement demand sensor with signal fusion
- [ ] Add drift detection algorithms
- [ ] Create baseline update mechanism
- [ ] Test with M5 data simulation
- [ ] Benchmark latency

### Phase 3: Anomaly Detection (Week 5-6)
- [ ] Implement statistical detectors (z-score, IQR)
- [ ] Add ML detectors (Isolation Forest)
- [ ] Implement business rule engine
- [ ] Create ensemble detection
- [ ] Build alert classification system
- [ ] Test false positive rates

### Phase 4: Short-Term Forecasting (Week 7-8)
- [ ] Implement Prophet forecaster
- [ ] Add SARIMAX forecaster
- [ ] Create ensemble forecasting
- [ ] Integrate with demand sensor
- [ ] Evaluate forecast accuracy

### Phase 5: Replenishment Automation (Week 9-10)
- [ ] Build trigger evaluation engine
- [ ] Implement order generation logic
- [ ] Create priority ranking system
- [ ] Add auto-approval rules
- [ ] Test end-to-end automation

### Phase 6: Dashboard Development (Week 11-13)
- [ ] Build Streamlit app structure
- [ ] Create monitoring dashboard
- [ ] Implement alerts dashboard
- [ ] Build replenishment queue interface
- [ ] Add what-if analysis tool
- [ ] Create KPI tracking dashboard
- [ ] Polish UI/UX

### Phase 7: Integration & Testing (Week 14-16)
- [ ] End-to-end system testing
- [ ] Performance optimization
- [ ] Comprehensive test suite
- [ ] Documentation and notebooks
- [ ] Demo video creation

## Required Libraries

```txt
# Core
pandas>=1.5.0
numpy>=1.23.0

# Forecasting
prophet>=1.1.0
statsmodels>=0.14.0

# Anomaly Detection
scikit-learn>=1.2.0
pyod>=1.1.0
tensorflow>=2.11.0  # For LSTM autoencoder

# Filtering
filterpy>=1.4.5  # Kalman filter

# Dashboard
streamlit>=1.20.0
plotly>=5.13.0

# Notifications
python-dotenv>=1.0.0
requests>=2.28.0

# Caching
redis>=4.5.0  # Optional, for production
```

## Key Concepts & Best Practices

### Real-Time System Design Principles

1. **Latency Minimization**:
   - Process data as it arrives (streaming)
   - Use in-memory caching
   - Optimize database queries
   - Batch where possible

2. **Fault Tolerance**:
   - Handle missing data gracefully
   - Implement fallback logic
   - Log errors without crashing
   - Use circuit breakers

3. **Scalability**:
   - Stateless processing where possible
   - Use message queues (Kafka in production)
   - Horizontal scaling
   - Load balancing

### Anomaly Detection Trade-offs

**Sensitivity vs. False Alarms**:
- **High sensitivity**: Catch all issues, but many false positives
- **Low sensitivity**: Miss issues, but few false positives
- **Solution**: Multi-level alerts (critical/warning/info) + human-in-loop

**Statistical vs. ML Methods**:
- **Statistical**: Fast, interpretable, no training needed
- **ML**: More accurate, adapts to patterns, requires training data
- **Recommendation**: Use ensemble approach

### Replenishment Automation Best Practices

1. **Human-in-Loop for High-Value**:
   - Auto-approve low-value routine orders
   - Review A items and non-routine orders
   - Set dollar thresholds

2. **Gradual Rollout**:
   - Start with C items only
   - Expand to B items after validation
   - A items last (with caution)

3. **Monitor Automation Quality**:
   - Track auto-approval accuracy
   - Review rejected orders
   - Adjust rules based on feedback

## Expected Results

### Performance Improvements
- **Stockout Reduction**: 25-30%
- **Response Time**: 90% faster (hours → minutes)
- **Automation Rate**: 80% of replenishment decisions
- **Inventory Reduction**: 15% lower average inventory
- **Service Level**: Improved from 94% to 98.5%

### Anomaly Detection Performance
- **Precision**: 80-85% (few false positives)
- **Recall**: 85-90% (catch most issues)
- **False Alarms**: < 5 per day
- **Detection Latency**: < 1 hour

### Automation Quality
- **Auto-Approval Accuracy**: 97%+
- **Order Timeliness**: 95% on-time
- **Cost Savings**: $1.2M annually (expedited shipping reduction)

## Troubleshooting

### Issue: Dashboard Slow to Load

**Solution**: Implement caching
```python
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_data():
    return expensive_query()
```

### Issue: Too Many False Alerts

**Solution**: Tune sensitivity or use multi-method voting
```python
# Require 2 out of 3 methods to agree
if sum([z_score_detect(), isolation_forest_detect(),
        business_rule_detect()]) >= 2:
    trigger_alert()
```

### Issue: Streaming Data Gaps

**Solution**: Implement imputation
```python
def handle_missing_data(data):
    # Forward fill for short gaps
    data = data.fillna(method='ffill', limit=3)

    # Use historical average for longer gaps
    data = data.fillna(data.rolling(24, min_periods=1).mean())

    return data
```

## Additional Resources

- **[README.md](README.md)** - Project overview
- **Streamlit Documentation**: https://docs.streamlit.io
- **Prophet Documentation**: https://facebook.github.io/prophet
- **PyOD (Anomaly Detection)**: https://pyod.readthedocs.io

## References

- Hawkins, D. M. (1980). *Identification of Outliers*
- Chandola, V., et al. (2009). *Anomaly Detection: A Survey*
- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale* (Prophet)
- Aggarwal, C. C. (2013). *Outlier Analysis*

## Contact

**Godson Kurishinkal**
- GitHub: [@GodsonKurishinkal](https://github.com/GodsonKurishinkal)
- LinkedIn: [linkedin.com/in/godsonkurishinkal](https://www.linkedin.com/in/godsonkurishinkal)
- Email: godson.kurishinkal+github@gmail.com
