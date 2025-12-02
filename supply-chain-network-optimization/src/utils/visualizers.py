"""Visualization utilities for supply chain networks and routes."""

import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go


class NetworkVisualizer:
    """Visualize supply chain networks on maps and charts."""

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.

        Parameters
        ----------
        style : str
            Matplotlib style to use
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_network_map(self, facilities: pd.DataFrame, stores: pd.DataFrame,
                        connections: Optional[pd.DataFrame] = None,
                        center: Optional[Tuple[float, float]] = None,
                        zoom: int = 5) -> folium.Map:
        """
        Create interactive map showing facilities, stores, and connections.

        Parameters
        ----------
        facilities : pd.DataFrame
            Facilities with columns: id, latitude, longitude, (status, capacity)
        stores : pd.DataFrame
            Stores with columns: id, latitude, longitude, (demand)
        connections : Optional[pd.DataFrame]
            Connections with columns: from_id, to_id, (flow)
        center : Optional[Tuple]
            Map center (lat, lon). If None, computed from data
        zoom : int
            Initial zoom level

        Returns
        -------
        folium.Map
            Interactive map
        """
        # Calculate center if not provided
        if center is None:
            all_lats = list(facilities['latitude']) + list(stores['latitude'])
            all_lons = list(facilities['longitude']) + list(stores['longitude'])
            center = (np.mean(all_lats), np.mean(all_lons))

        # Create map
        m = folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')

        # Add facilities
        for _, facility in facilities.iterrows():
            is_open = facility.get('status', 'open') == 'open'
            color = 'green' if is_open else 'gray'

            tooltip = f"Facility: {facility['id']}"
            if 'capacity' in facility:
                tooltip += f"<br>Capacity: {facility['capacity']}"

            folium.CircleMarker(
                location=[facility['latitude'], facility['longitude']],
                radius=10,
                popup=tooltip,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(m)

        # Add stores
        for _, store in stores.iterrows():
            tooltip = f"Store: {store['id']}"
            if 'demand' in store:
                tooltip += f"<br>Demand: {store['demand']}"

            folium.CircleMarker(
                location=[store['latitude'], store['longitude']],
                radius=5,
                popup=tooltip,
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.5,
                weight=1
            ).add_to(m)

        # Add connections
        if connections is not None:
            # Create lookup dictionaries
            fac_coords = facilities.set_index('id')[['latitude', 'longitude']].to_dict('index')
            store_coords = stores.set_index('id')[['latitude', 'longitude']].to_dict('index')
            all_coords = {**fac_coords, **store_coords}

            for _, conn in connections.iterrows():
                from_id = conn['from_id']
                to_id = conn['to_id']

                if from_id in all_coords and to_id in all_coords:
                    from_loc = [all_coords[from_id]['latitude'],
                               all_coords[from_id]['longitude']]
                    to_loc = [all_coords[to_id]['latitude'],
                             all_coords[to_id]['longitude']]

                    # Line width based on flow
                    weight = 2
                    if 'flow' in conn and conn['flow'] > 0:
                        weight = min(8, max(1, conn['flow'] / 100))

                    folium.PolyLine(
                        locations=[from_loc, to_loc],
                        color='red',
                        weight=weight,
                        opacity=0.6
                    ).add_to(m)

        return m

    def plot_cost_tradeoff(self, results: List[Dict],
                          x_metric: str = 'num_facilities',
                          y_metric: str = 'total_cost') -> plt.Figure:
        """
        Plot cost trade-off curve.

        Parameters
        ----------
        results : List[Dict]
            List of optimization results with metrics
        x_metric : str
            Metric for x-axis (e.g., 'num_facilities')
        y_metric : str
            Metric for y-axis (e.g., 'total_cost')

        Returns
        -------
        plt.Figure
            Trade-off curve plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        df = pd.DataFrame(results)

        ax.plot(df[x_metric], df[y_metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate points
        for _, row in df.iterrows():
            ax.annotate(
                f"{row[y_metric]:.1f}",
                (row[x_metric], row[y_metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )

        plt.tight_layout()
        return fig

    def plot_facility_utilization(self, utilization: Dict[str, float]) -> plt.Figure:
        """
        Plot facility utilization bar chart.

        Parameters
        ----------
        utilization : Dict[str, float]
            Mapping of facility_id to utilization percentage (0-1)

        Returns
        -------
        plt.Figure
            Bar chart of utilization
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        facilities = list(utilization.keys())
        util_pct = [v * 100 for v in utilization.values()]

        colors = ['green' if u >= 70 else 'orange' if u >= 50 else 'red'
                 for u in util_pct]

        ax.bar(facilities, util_pct, color=colors, alpha=0.7)
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.set_xlabel('Facility', fontsize=12)
        ax.set_ylabel('Utilization (%)', fontsize=12)
        ax.set_title('Facility Utilization', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Rotate x-labels if many facilities
        if len(facilities) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_sankey_flow(self, flows: pd.DataFrame) -> go.Figure:
        """
        Create Sankey diagram showing flow through network.

        Parameters
        ----------
        flows : pd.DataFrame
            DataFrame with columns: source, target, value

        Returns
        -------
        go.Figure
            Plotly Sankey diagram
        """
        # Create node labels
        all_nodes = list(set(flows['source'].unique()) | set(flows['target'].unique()))
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}

        # Map sources and targets to indices
        sources = [node_dict[s] for s in flows['source']]
        targets = [node_dict[t] for t in flows['target']]
        values = flows['value'].tolist()

        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_nodes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])

        fig.update_layout(
            title_text="Network Flow Distribution",
            font_size=10,
            height=600
        )

        return fig


class RouteVisualizer:
    """Visualize vehicle routes and delivery schedules."""

    def __init__(self):
        """Initialize route visualizer."""
        pass

    def plot_routes_on_map(self, depot: Tuple[float, float],
                          routes: List[List[Dict]],
                          zoom: int = 11) -> folium.Map:
        """
        Plot multiple vehicle routes on a map.

        Parameters
        ----------
        depot : Tuple[float, float]
            Depot location (latitude, longitude)
        routes : List[List[Dict]]
            List of routes, each route is a list of stops with:
            {id, latitude, longitude, arrival_time}
        zoom : int
            Map zoom level

        Returns
        -------
        folium.Map
            Interactive map with routes
        """
        m = folium.Map(location=depot, zoom_start=zoom)

        # Add depot
        folium.Marker(
            location=depot,
            popup="Depot",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)

        # Route colors
        colors = ['blue', 'green', 'purple', 'orange', 'darkred',
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']

        # Plot each route
        for route_idx, route in enumerate(routes):
            color = colors[route_idx % len(colors)]

            # Route path
            path = [depot]
            for stop in route:
                path.append((stop['latitude'], stop['longitude']))
            path.append(depot)  # Return to depot

            # Draw route line
            folium.PolyLine(
                locations=path,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Route {route_idx + 1}"
            ).add_to(m)

            # Add stop markers
            for stop_idx, stop in enumerate(route):
                folium.CircleMarker(
                    location=[stop['latitude'], stop['longitude']],
                    radius=7,
                    popup=f"Route {route_idx + 1}<br>Stop {stop_idx + 1}<br>{stop['id']}<br>ETA: {stop.get('arrival_time', 'N/A')}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=2
                ).add_to(m)

        return m

    def plot_route_schedule(self, routes: List[Dict]) -> plt.Figure:
        """
        Plot Gantt chart of route schedules.

        Parameters
        ----------
        routes : List[Dict]
            List of routes with: vehicle_id, start_time, end_time, duration

        Returns
        -------
        plt.Figure
            Gantt chart
        """
        fig, ax = plt.subplots(figsize=(12, max(6, len(routes) * 0.5)))

        for idx, route in enumerate(routes):
            start = route['start_time']
            duration = route['duration']

            ax.barh(idx, duration, left=start, height=0.6,
                   label=route.get('vehicle_id', f'Route {idx + 1}'))

        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Route', fontsize=12)
        ax.set_title('Route Schedule', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(routes)))
        ax.set_yticklabels([route.get('vehicle_id', f'Route {i + 1}')
                            for i, route in enumerate(routes)])
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_route_metrics(self, metrics: pd.DataFrame) -> plt.Figure:
        """
        Plot route performance metrics.

        Parameters
        ----------
        metrics : pd.DataFrame
            Metrics with columns: route_id, distance, duration, stops, utilization

        Returns
        -------
        plt.Figure
            Multi-panel metrics plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distance
        axes[0, 0].bar(metrics['route_id'], metrics['distance'])
        axes[0, 0].set_title('Distance per Route')
        axes[0, 0].set_ylabel('Miles')
        axes[0, 0].grid(True, alpha=0.3)

        # Duration
        axes[0, 1].bar(metrics['route_id'], metrics['duration'], color='orange')
        axes[0, 1].set_title('Duration per Route')
        axes[0, 1].set_ylabel('Hours')
        axes[0, 1].grid(True, alpha=0.3)

        # Stops
        axes[1, 0].bar(metrics['route_id'], metrics['stops'], color='green')
        axes[1, 0].set_title('Stops per Route')
        axes[1, 0].set_ylabel('Number of Stops')
        axes[1, 0].grid(True, alpha=0.3)

        # Utilization
        axes[1, 1].bar(metrics['route_id'], metrics['utilization'] * 100, color='purple')
        axes[1, 1].set_title('Vehicle Utilization')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
