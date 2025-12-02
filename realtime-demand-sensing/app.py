"""
Real-Time Demand Sensing Dashboard

Interactive Streamlit dashboard for monitoring demand, detecting anomalies,
and managing replenishment decisions.

Uses:
- StreamSimulator for realistic data generation
- AnomalyDetector for multi-method anomaly detection
- ShortTermForecaster for demand forecasting
- ReplenishmentEngine for automated ordering

Author: Godson Kurishinkal
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Import modules
try:
    from src.utils import StreamSimulator
    from src.sensing import DemandSensor
    from src.detection import AnomalyDetector, AlertManager, AnomalySeverity
    from src.forecasting import ShortTermForecaster
    from src.replenishment import ReplenishmentEngine, InventoryPosition
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Real-Time Demand Sensing",
    page_icon="📊",
    layout="wide"
)


def generate_sample_data(days=30, n_products=5):
    """Generate sample data using StreamSimulator or fallback."""
    if MODULES_AVAILABLE:
        # Use our StreamSimulator
        all_data = []
        for i in range(n_products):
            simulator = StreamSimulator(
                product_id=f"PROD_{i+1:03d}",
                base_demand=np.random.uniform(50, 150),
                anomaly_probability=0.02,
                seed=42 + i
            )
            product_data = simulator.generate_historical(n_days=days)
            all_data.append(product_data)
        
        df = pd.concat(all_data, ignore_index=True)
    else:
        # Fallback to simple generation
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        all_data = []
        
        for i in range(n_products):
            base_demand = np.random.uniform(50, 150)
            base = base_demand + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))
            hour_of_day = dates.hour
            hourly_pattern = 1 + 0.3 * np.sin((hour_of_day - 14) * np.pi / 12)
            demand = base * hourly_pattern + np.random.normal(0, 5, len(dates))
            
            # Insert anomalies
            anomaly_indices = np.random.choice(len(dates), size=5, replace=False)
            demand[anomaly_indices] *= np.random.choice([2.5, 0.3], size=5)
            
            product_df = pd.DataFrame({
                'timestamp': dates,
                'product_id': f"PROD_{i+1:03d}",
                'sales': np.maximum(0, demand)
            })
            all_data.append(product_df)
        
        df = pd.concat(all_data, ignore_index=True)
    
    return df


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_data():
    """Load and cache data."""
    return generate_sample_data(days=14, n_products=5)


def main():
    """Main dashboard."""
    
    # Header
    st.title("📊 Real-Time Demand Sensing Dashboard")
    st.markdown("Monitor demand signals, detect anomalies, and manage replenishment in real-time")
    
    # Module status
    if not MODULES_AVAILABLE:
        st.warning("⚠️ Some modules not available. Using fallback data generation.")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", 1, 60, 15)
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold (σ)", 2.0, 4.0, 3.0, 0.5)
    
    selected_product = st.sidebar.selectbox(
        "Select Product",
        options=[f"PROD_{i:03d}" for i in range(1, 6)],
        index=0
    )
    
    # Load data
    data = load_data()
    product_data = data[data['product_id'] == selected_product].copy()
    
    # Calculate metrics
    recent_sales = product_data.tail(24)['sales'].mean()
    baseline = product_data['sales'].rolling(window=168).mean().iloc[-1]  # 7-day baseline
    change_pct = ((recent_sales - baseline) / baseline) * 100 if baseline > 0 else 0
    
    # Simulate inventory
    daily_sales = recent_sales * 24
    initial_inventory = 5000
    cumulative_sales = product_data['sales'].cumsum()
    current_inventory = max(100, initial_inventory - cumulative_sales.iloc[-1] * 0.5)
    days_of_stock = current_inventory / daily_sales if daily_sales > 0 else 999
    
    # Detect anomalies
    product_data['zscore'] = (product_data['sales'] - product_data['sales'].mean()) / product_data['sales'].std()
    anomalies_24h = len(product_data.tail(24)[np.abs(product_data.tail(24)['zscore']) > anomaly_threshold])
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Demand Rate",
            f"{recent_sales:.0f} units/hr",
            f"{change_pct:+.1f}%"
        )
    
    with col2:
        status = "🚨 Low" if days_of_stock < 3 else "✅ OK"
        st.metric(
            "Current Inventory",
            f"{current_inventory:.0f} units",
            status
        )
    
    with col3:
        health = "Critical" if days_of_stock < 1.5 else "Warning" if days_of_stock < 3 else "Healthy"
        st.metric(
            "Days of Stock",
            f"{days_of_stock:.1f} days",
            health
        )
    
    with col4:
        alert = "⚠️ Check" if anomalies_24h > 0 else "✅ OK"
        st.metric(
            "Anomalies (24h)",
            anomalies_24h,
            alert
        )
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand Monitoring", "🔍 Anomaly Detection", "🎯 Forecasting", "📦 Replenishment"])
    
    with tab1:
        st.subheader(f"Demand Trend - {selected_product}")
        
        # Demand chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=product_data['timestamp'],
            y=product_data['sales'],
            mode='lines',
            name='Actual Sales',
            line=dict(color='blue', width=1)
        ))
        
        # Rolling average
        product_data['rolling_avg'] = product_data['sales'].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=product_data['timestamp'],
            y=product_data['rolling_avg'],
            mode='lines',
            name='24h Moving Avg',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Sales (units/hour)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Inventory simulation chart
        st.subheader("Simulated Inventory Level")
        inventory_series = initial_inventory - product_data['sales'].cumsum() * 0.5
        inventory_series = np.maximum(100, inventory_series)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=product_data['timestamp'],
            y=inventory_series,
            mode='lines',
            fill='tozeroy',
            name='Inventory',
            line=dict(color='green')
        ))
        
        # Add threshold lines
        safety_stock = 1000
        reorder_point = 2000
        fig2.add_hline(y=safety_stock, line_dash="dash", line_color="red", 
                      annotation_text="Safety Stock")
        fig2.add_hline(y=reorder_point, line_dash="dash", line_color="orange",
                      annotation_text="Reorder Point")
        
        fig2.update_layout(
            height=300,
            xaxis_title="Time",
            yaxis_title="Inventory (units)",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Anomaly Detection")
        
        # Detect anomalies using our detector if available
        if MODULES_AVAILABLE:
            detector = AnomalyDetector(z_threshold=anomaly_threshold)
            detector.fit(product_data.rename(columns={'sales': 'value'}))
            detected = detector.detect(product_data.rename(columns={'sales': 'value'}))
            
            anomaly_times = [a.timestamp for a in detected]
            anomalies = product_data[product_data['timestamp'].isin(anomaly_times)].copy()
        else:
            # Fallback to simple z-score
            anomalies = product_data[np.abs(product_data['zscore']) > anomaly_threshold].copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly chart
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=product_data['timestamp'],
                y=product_data['sales'],
                mode='lines',
                name='Sales',
                line=dict(color='lightblue')
            ))
            
            if len(anomalies) > 0:
                fig3.add_trace(go.Scatter(
                    x=anomalies['timestamp'],
                    y=anomalies['sales'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig3.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Sales",
                hovermode='x unified'
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.markdown("### Recent Anomalies")
            if MODULES_AVAILABLE and detected:
                for anomaly in detected[-5:]:
                    severity_color = "🔴" if anomaly.severity == AnomalySeverity.CRITICAL else "🟡" if anomaly.severity == AnomalySeverity.WARNING else "🔵"
                    st.warning(f"**{severity_color} {anomaly.anomaly_type.value.upper()}**  \n{anomaly.timestamp.strftime('%Y-%m-%d %H:%M')}  \n{anomaly.message}")
            elif len(anomalies) > 0:
                for _, row in anomalies.tail(5).iterrows():
                    anomaly_type = "📈 SPIKE" if row['zscore'] > 0 else "📉 DROP"
                    st.warning(f"**{anomaly_type}**  \n{row['timestamp'].strftime('%Y-%m-%d %H:%M')}  \nSales: {row['sales']:.0f} units (z={row['zscore']:.2f})")
            else:
                st.success("No anomalies detected in the selected period")
    
    with tab3:
        st.subheader("Short-Term Forecast")
        
        # Prepare forecast data
        forecast_input = product_data[['timestamp', 'sales']].rename(columns={'sales': 'value'})
        
        if MODULES_AVAILABLE:
            forecaster = ShortTermForecaster(model='ensemble')
            forecaster.fit(forecast_input)
            forecast_df = forecaster.forecast(horizon=48)  # 48 hours
        else:
            # Simple EWM fallback
            alpha = 0.3
            product_data['forecast'] = product_data['sales'].ewm(alpha=alpha, adjust=False).mean()
            last_value = product_data['forecast'].iloc[-1]
            
            future_dates = pd.date_range(
                start=product_data['timestamp'].iloc[-1] + timedelta(hours=1),
                periods=48, freq='H'
            )
            forecast_df = pd.DataFrame({
                'timestamp': future_dates,
                'forecast': [last_value] * len(future_dates),
                'lower_bound': [last_value * 0.8] * len(future_dates),
                'upper_bound': [last_value * 1.2] * len(future_dates)
            })
        
        # Chart
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=product_data['timestamp'].tail(7*24),
            y=product_data['sales'].tail(7*24),
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        fig4.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='orange', dash='dash')
        ))
        
        # Add confidence interval
        if 'lower_bound' in forecast_df.columns:
            fig4.add_trace(go.Scatter(
                x=pd.concat([forecast_df['timestamp'], forecast_df['timestamp'][::-1]]),
                y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))
        
        fig4.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Sales",
            hovermode='x unified'
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_24h = forecast_df['forecast'].head(24).sum()
            st.metric("Next 24h Forecast", f"{forecast_24h:.0f} units")
        with col2:
            forecast_48h = forecast_df['forecast'].sum()
            st.metric("Next 48h Forecast", f"{forecast_48h:.0f} units")
        with col3:
            # Calculate MAE on last 24h
            if 'forecast' in product_data.columns:
                mae = np.abs(product_data['sales'].tail(24) - product_data['forecast'].tail(24)).mean()
            else:
                mae = product_data['sales'].tail(24).std() * 0.5  # Estimate
            st.metric("Forecast MAE", f"{mae:.1f} units")
    
    with tab4:
        st.subheader("Replenishment Queue")
        
        # Generate product inventory data
        products = pd.DataFrame({
            'product_id': [f'PROD_{i:03d}' for i in range(1, 6)],
            'on_hand': [500, 1200, 300, 800, 150],
            'on_order': [0, 200, 0, 0, 100],
            'allocated': [50, 100, 30, 80, 20],
            'safety_stock': [200, 300, 250, 180, 160],
            'reorder_point': [400, 500, 400, 350, 300],
            'avg_daily_demand': [100, 150, 120, 90, 80],
            'lead_time_days': [3, 5, 3, 4, 3],
            'unit_cost': [15.0, 25.0, 18.0, 12.0, 22.0]
        })
        
        if MODULES_AVAILABLE:
            # Use ReplenishmentEngine
            engine = ReplenishmentEngine(
                target_dos=14,
                auto_approve_limit=500,
                auto_approve_emergency=True
            )
            engine.update_positions(products)
            results = engine.run_cycle()
            
            orders_df = engine.orders_to_dataframe()
            summary = engine.get_summary()
        else:
            # Simple calculations
            products['available'] = products['on_hand'] + products['on_order'] - products['allocated']
            products['days_of_stock'] = products['available'] / products['avg_daily_demand']
            orders_df = pd.DataFrame()
            summary = {}
        
        # Display products table
        products['available'] = products['on_hand'] + products['on_order'] - products['allocated']
        products['days_of_stock'] = products['available'] / products['avg_daily_demand']
        products['status'] = products['days_of_stock'].apply(
            lambda x: '🚨 Critical' if x < 1.5 else '⚠️ Warning' if x < 3 else '✅ OK'
        )
        products['action'] = products['days_of_stock'].apply(
            lambda x: 'Order Now' if x < 1.5 else 'Schedule' if x < 3 else 'Monitor'
        )
        
        # Display styled table
        display_cols = ['product_id', 'on_hand', 'available', 'days_of_stock', 'status', 'action']
        st.dataframe(
            products[display_cols].style.apply(
                lambda x: ['background-color: #ffcccc' if x['status'] == '🚨 Critical' 
                          else 'background-color: #fff4cc' if x['status'] == '⚠️ Warning'
                          else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical = len(products[products['status'] == '🚨 Critical'])
            st.metric("Critical Items", critical)
        with col2:
            warning = len(products[products['status'] == '⚠️ Warning'])
            st.metric("Warning Items", warning)
        with col3:
            healthy = len(products[products['status'] == '✅ OK'])
            st.metric("Healthy Items", healthy)
        with col4:
            if not orders_df.empty:
                total_value = orders_df['total_cost'].sum()
                st.metric("Pending Orders", f"${total_value:,.0f}")
            else:
                st.metric("Pending Orders", "$0")
        
        # Show generated orders
        if not orders_df.empty:
            st.subheader("Generated Replenishment Orders")
            st.dataframe(
                orders_df[['id', 'product_id', 'quantity', 'priority', 'status', 'total_cost']],
                use_container_width=True,
                hide_index=True
            )
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.caption(f"Auto-refresh: {refresh_interval} min")
    with col3:
        status = "✅ All modules loaded" if MODULES_AVAILABLE else "⚠️ Fallback mode"
        st.caption(status)


if __name__ == "__main__":
    main()
