"""Graph utilities for network optimization."""

import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional


class NetworkGraphBuilder:
    """Build and analyze supply chain network graphs."""

    def __init__(self):
        """Initialize network graph builder."""
        self.graph = nx.DiGraph()

    def add_facilities(self, facilities: pd.DataFrame, node_type: str = 'facility'):
        """
        Add facility nodes to the graph.

        Parameters
        ----------
        facilities : pd.DataFrame
            DataFrame with facility information (id, capacity, fixed_cost, etc.)
        node_type : str
            Type label for these nodes
        """
        for _, facility in facilities.iterrows():
            self.graph.add_node(
                facility['id'],
                node_type=node_type,
                capacity=facility.get('capacity', None),
                fixed_cost=facility.get('fixed_cost', None),
                latitude=facility.get('latitude', None),
                longitude=facility.get('longitude', None)
            )

    def add_stores(self, stores: pd.DataFrame):
        """
        Add store/customer nodes to the graph.

        Parameters
        ----------
        stores : pd.DataFrame
            DataFrame with store information (id, demand, etc.)
        """
        for _, store in stores.iterrows():
            self.graph.add_node(
                store['id'],
                node_type='store',
                demand=store.get('demand', None),
                latitude=store.get('latitude', None),
                longitude=store.get('longitude', None)
            )

    def add_connections(self, connections: pd.DataFrame,
                       distance_col: str = 'distance',
                       cost_col: str = 'cost',
                       flow_col: Optional[str] = None):
        """
        Add edges between nodes.

        Parameters
        ----------
        connections : pd.DataFrame
            DataFrame with columns: from_id, to_id, distance, cost, flow (optional)
        distance_col : str
            Column name for distance
        cost_col : str
            Column name for cost
        flow_col : Optional[str]
            Column name for flow quantity (if available)
        """
        for _, conn in connections.iterrows():
            edge_data = {
                'distance': conn[distance_col],
                'cost': conn[cost_col]
            }
            if flow_col and flow_col in conn:
                edge_data['flow'] = conn[flow_col]

            self.graph.add_edge(
                conn['from_id'],
                conn['to_id'],
                **edge_data
            )

    def build_bipartite_graph(self, facilities: pd.DataFrame, stores: pd.DataFrame,
                             distance_matrix: pd.DataFrame,
                             cost_per_mile: float = 0.50) -> nx.DiGraph:
        """
        Build a bipartite graph of facilities and stores.

        Parameters
        ----------
        facilities : pd.DataFrame
            Facility data
        stores : pd.DataFrame
            Store data
        distance_matrix : pd.DataFrame
            Distance matrix
        cost_per_mile : float
            Transportation cost per mile

        Returns
        -------
        nx.DiGraph
            Bipartite directed graph
        """
        # Reset graph
        self.graph = nx.DiGraph()

        # Add facilities
        self.add_facilities(facilities, node_type='facility')

        # Add stores
        self.add_stores(stores)

        # Add all possible connections
        connections = []
        for _, facility in facilities.iterrows():
            for _, store in stores.iterrows():
                fac_id = facility['id']
                store_id = store['id']

                if fac_id in distance_matrix.index and store_id in distance_matrix.columns:
                    distance = distance_matrix.loc[fac_id, store_id]
                    cost = distance * cost_per_mile

                    connections.append({
                        'from_id': fac_id,
                        'to_id': store_id,
                        'distance': distance,
                        'cost': cost
                    })

        connections_df = pd.DataFrame(connections)
        self.add_connections(connections_df)

        return self.graph

    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph:
        """Extract subgraph containing only specified nodes."""
        return self.graph.subgraph(node_ids).copy()

    def shortest_path(self, from_id: str, to_id: str,
                     weight: str = 'distance') -> Tuple[List[str], float]:
        """
        Find shortest path between two nodes.

        Returns
        -------
        Tuple of (path, total_weight)
        """
        try:
            path = nx.shortest_path(self.graph, from_id, to_id, weight=weight)
            length = nx.shortest_path_length(self.graph, from_id, to_id, weight=weight)
            return path, length
        except nx.NetworkXNoPath:
            return [], float('inf')

    def calculate_flow_metrics(self) -> Dict:
        """Calculate network flow metrics."""
        metrics = {}

        # Total flow
        total_flow = sum(
            data.get('flow', 0)
            for _, _, data in self.graph.edges(data=True)
        )
        metrics['total_flow'] = total_flow

        # Average flow per edge
        edges_with_flow = sum(
            1 for _, _, data in self.graph.edges(data=True)
            if data.get('flow', 0) > 0
        )
        metrics['avg_flow_per_edge'] = total_flow / edges_with_flow if edges_with_flow > 0 else 0

        # Total cost
        total_cost = sum(
            data.get('cost', 0) * data.get('flow', 0)
            for _, _, data in self.graph.edges(data=True)
        )
        metrics['total_cost'] = total_cost

        # Facility utilization
        facility_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == 'facility'
        ]

        for facility in facility_nodes:
            outflow = sum(
                data.get('flow', 0)
                for _, _, data in self.graph.out_edges(facility, data=True)
            )
            capacity = self.graph.nodes[facility].get('capacity', 0)
            utilization = outflow / capacity if capacity > 0 else 0
            metrics[f'{facility}_utilization'] = utilization

        return metrics

    def get_facility_assignments(self) -> Dict[str, List[str]]:
        """
        Get which stores are assigned to which facilities.

        Returns
        -------
        Dict mapping facility_id to list of store_ids
        """
        assignments = {}

        facility_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == 'facility'
        ]

        for facility in facility_nodes:
            # Get all stores connected from this facility with positive flow
            stores = [
                target for _, target, data in self.graph.out_edges(facility, data=True)
                if data.get('flow', 0) > 0
            ]
            assignments[facility] = stores

        return assignments

    def calculate_centrality(self) -> pd.DataFrame:
        """Calculate various centrality measures for nodes."""
        centrality_measures = {}

        # Degree centrality
        centrality_measures['degree'] = nx.degree_centrality(self.graph)

        # Betweenness centrality
        centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph)

        # Closeness centrality (for strongly connected components)
        if nx.is_strongly_connected(self.graph):
            centrality_measures['closeness'] = nx.closeness_centrality(self.graph)

        return pd.DataFrame(centrality_measures)

    def export_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export graph to node and edge DataFrames.

        Returns
        -------
        Tuple of (nodes_df, edges_df)
        """
        # Nodes
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            node_dict = {'id': node}
            node_dict.update(data)
            nodes_data.append(node_dict)
        nodes_df = pd.DataFrame(nodes_data)

        # Edges
        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edge_dict = {
                'from_id': source,
                'to_id': target
            }
            edge_dict.update(data)
            edges_data.append(edge_dict)
        edges_df = pd.DataFrame(edges_data)

        return nodes_df, edges_df
