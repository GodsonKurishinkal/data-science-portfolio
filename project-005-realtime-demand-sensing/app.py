"""
Real-Time Demand Sensing Dashboard

Interactive Streamlit dashboard for monitoring demand, detecting anomalies,
and managing replenishment decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Demand Sensing Dashboard",
    page_icon="📊",
    layout="wide"
)


def generate_sample_data(days=30):
    """Generate sample data for dashboard."""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    base_demand = 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7))
    hour_of_day = dates.hour
    hourly_pattern = 1 + 0.3 * np.sin((hour_of_day - 14) * np.pi / 12)
    demand = base_demand * hourly_pattern + np.random.normal(0, 5, len(dates))
    
    # Insert anomalies
    anomaly_indices = np.random.choice(len(dates), size=8, replace=False)
    demand[anomaly_indices] *= np.random.choice([2.5, 0.3], size=8)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'sales': np.maximum(0, demand),
        'inventory': 5000 - np.cumsum(demand) * 0.8
    })
    df['inventory'] = np.maximum(100, df['inventory'])
    
    return df


@st.cache_data
def load_data():
    """Load and cache data."""
    return generate_sample_data(days=30)


def main():
    """Main dashboard."""
    
    # Header
    st.title("📊 Real-Time Demand Sensing Dashboard")
    st.markdown("Monitor demand signals, detect anomalies, and manage replenishment in real-time")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", 1, 60, 15)
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold (σ)", 2.0, 4.0, 3.0, 0.5)
    
    # Load data
    data = load_data()
    
    # Calculate metrics
    recent_sales = data.tail(24)['sales'].mean()
    baseline = data['sales'].rolling(window=24).mean().iloc[-1]
    change_pct = ((recent_sales - baseline) / baseline) * 100 if baseline > 0 else 0
    current_inventory = data['inventory'].iloc[-1]
    days_of_stock = current_inventory / (recent_sales * 24) if recent_sales > 0 else 999
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Demand Rate",
            f"{recent_sales:.0f} units/hr",
            f"{change_pct:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Inventory",
            f"{current_inventory:.0f} units",
            "🚨 Low" if days_of_stock < 3 else "✅ OK"
        )
    
    with col3:
        st.metric(
            "Days of Stock",
            f"{days_of_stock:.1f} days",
            "Critical" if days_of_stock < 1.5 else "Warning" if days_of_stock < 3 else "Healthy"
        )
    
    with col4:
        data['zscore'] = (data['sales'] - data['sales'].mean()) / data['sales'].std()
        anomalies_24h = len(data.tail(24)[np.abs(data.tail(24)['zscore']) > anomaly_threshold])
        st.metric(
            "Anomalies (24h)",
            anomalies_24h,
            "⚠️ Check" if anomalies_24h > 0 else "✅ OK"
        )
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand Monitoring", "🔍 Anomaly Detection", "🎯 Forecasting", "📦 Replenishment"])
    
    with tab1:
        st.subheader("Demand Trend")
        
        # Demand chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['sales'],
            mode='lines',
            name='Actual Sales',
            line=dict(color='blue', width=1)
        ))
        
        # Rolling average
        data['rolling_avg'] = data['sales'].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['rolling_avg'],
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
        
        # Inventory chart
        st.subheader("Inventory Level")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['inventory'],
            mode='lines',
            fill='tozeroy',
            name='Inventory',
            line=dict(color='green')
        ))
        
        # Add threshold lines
        fig2.add_hline(y=1000, line_dash="dash", line_color="red", 
                      annotation_text="Critical Level")
        fig2.add_hline(y=2000, line_dash="dash", line_color="orange",
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
        
        # Detect anomalies
        anomalies = data[np.abs(data['zscore']) > anomaly_threshold].copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly chart
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['sales'],
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
            if len(anomalies) > 0:
                for _, row in anomalies.tail(5).iterrows():
                    anomaly_type = "📈 SPIKE" if row['zscore'] > 0 else "📉 DROP"
                    st.warning(f"**{anomaly_type}**  \n{row['timestamp'].strftime('%Y-%m-%d %H:%M')}  \nSales: {row['sales']:.0f} units (z={row['zscore']:.2f})")
            else:
                st.success("No anomalies detected in the selected period")
    
    with tab3:
        st.subheader("Short-Term Forecast")
        
        # Simple forecast
        alpha = 0.3
        data['forecast'] = data['sales'].ewm(alpha=alpha, adjust=False).mean()
        last_value = data['forecast'].iloc[-1]
        
        # Generate 7-day forecast
        future_dates = pd.date_range(start=data['timestamp'].iloc[-1] + timedelta(hours=1),
                                    periods=7*24, freq='H')
        future_values = [last_value] * len(future_dates)
        
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': future_values
        })
        
        # Chart
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=data['timestamp'].tail(7*24),
            y=data['sales'].tail(7*24),
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
        
        fig4.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Sales",
            hovermode='x unified'
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Next 24h Forecast", f"{last_value*24:.0f} units")
        with col2:
            st.metric("Next 7d Forecast", f"{last_value*24*7:.0f} units")
        with col3:
            mae = np.abs(data['sales'].tail(7*24) - data['forecast'].tail(7*24)).mean()
            st.metric("Forecast MAE", f"{mae:.1f} units")
    
    with tab4:
        st.subheader("Replenishment Queue")
        
        # Generate product list
        products = pd.DataFrame({
            'Product': [f'PROD_{i:03d}' for i in range(1, 11)],
            'Inventory': [500, 1200, 300, 800, 2000, 150, 600, 900, 400, 1100],
            'Daily Sales': [100, 150, 120, 90, 200, 80, 110, 95, 130, 85],
            'Safety Stock': [200, 300, 250, 180, 400, 160, 220, 190, 260, 170]
        })
        
        products['Days of Stock'] = products['Inventory'] / products['Daily Sales']
        products['Status'] = products['Days of Stock'].apply(
            lambda x: '🚨 Critical' if x < 1.5 else '⚠️ Warning' if x < 3 else '✅ OK'
        )
        products['Action'] = products['Days of Stock'].apply(
            lambda x: 'Order Now' if x < 1.5 else 'Schedule' if x < 3 else 'Monitor'
        )
        
        # Filter
        status_filter = st.multiselect(
            "Filter by Status",
            ['🚨 Critical', '⚠️ Warning', '✅ OK'],
            default=['🚨 Critical', '⚠️ Warning']
        )
        
        filtered = products[products['Status'].isin(status_filter)]
        
        # Display table
        st.dataframe(
            filtered.style.apply(
                lambda x: ['background-color: #ffcccc' if x['Status'] == '🚨 Critical' 
                          else 'background-color: #fff4cc' if x['Status'] == '⚠️ Warning'
                          else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            critical = len(products[products['Status'] == '🚨 Critical'])
            st.metric("Critical Items", critical)
        with col2:
            warning = len(products[products['Status'] == '⚠️ Warning'])
            st.metric("Warning Items", warning)
        with col3:
            healthy = len(products[products['Status'] == '✅ OK'])
            st.metric("Healthy Items", healthy)
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {refresh_interval} min")


if __name__ == "__main__":
    main()
