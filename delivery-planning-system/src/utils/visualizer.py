"""Visualization utilities for packing and routes."""
from typing import List, Dict
import json


class PackingVisualizer:
    """
    Visualizer for 3D bin packing results.
    
    Generates matplotlib 3D plots and Plotly interactive visualizations.
    """
    
    def __init__(self, container_dims: Dict, boxes: List[Dict]):
        """
        Initialize visualizer.
        
        Args:
            container_dims: Container dimensions {length, width, height}
            boxes: List of box data with positions and dimensions
        """
        self.container = container_dims
        self.boxes = boxes
    
    def to_plotly_data(self) -> Dict:
        """
        Generate Plotly 3D scatter data.
        
        Returns:
            Dictionary with Plotly trace data
        """
        traces = []
        
        # Container outline
        L, W, H = self.container["length"], self.container["width"], self.container["height"]
        
        # Container edges
        container_trace = {
            "type": "scatter3d",
            "mode": "lines",
            "x": [0, L, L, 0, 0, 0, L, L, 0, 0, L, L, L, L, 0, 0],
            "y": [0, 0, W, W, 0, 0, 0, W, W, 0, 0, 0, W, W, W, W],
            "z": [0, 0, 0, 0, 0, H, H, H, H, H, H, 0, 0, H, H, 0],
            "line": {"color": "gray", "width": 2},
            "name": "Container",
        }
        traces.append(container_trace)
        
        # Color palette for boxes
        colors = [
            "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", 
            "#E91E63", "#00BCD4", "#FFEB3B", "#795548",
        ]
        
        # Add boxes
        for i, box in enumerate(self.boxes):
            color = colors[i % len(colors)]
            box_trace = self._create_box_mesh(box, color)
            traces.append(box_trace)
        
        return {
            "data": traces,
            "layout": {
                "scene": {
                    "xaxis": {"title": "Length (cm)", "range": [0, L * 1.1]},
                    "yaxis": {"title": "Width (cm)", "range": [0, W * 1.1]},
                    "zaxis": {"title": "Height (cm)", "range": [0, H * 1.1]},
                    "aspectmode": "data",
                },
                "title": "3D Bin Packing Visualization",
            },
        }
    
    def _create_box_mesh(self, box: Dict, color: str) -> Dict:
        """Create a 3D mesh trace for a box."""
        pos = box["position"]
        dims = box["dimensions"]
        
        x, y, z = pos["x"], pos["y"], pos["z"]
        l, w, h = dims["length"], dims["width"], dims["height"]
        
        # 8 vertices of the box
        vertices_x = [x, x+l, x+l, x, x, x+l, x+l, x]
        vertices_y = [y, y, y+w, y+w, y, y, y+w, y+w]
        vertices_z = [z, z, z, z, z+h, z+h, z+h, z+h]
        
        # 12 triangular faces (2 per box face)
        i = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
        j = [1, 2, 5, 6, 1, 4, 2, 5, 3, 4, 2, 6]
        k = [2, 3, 6, 7, 4, 5, 5, 6, 4, 7, 6, 7]
        
        return {
            "type": "mesh3d",
            "x": vertices_x,
            "y": vertices_y,
            "z": vertices_z,
            "i": i,
            "j": j,
            "k": k,
            "color": color,
            "opacity": 0.7,
            "name": f"Box {box.get('id', '')} (Seq: {box.get('sequence', '')})",
            "hoverinfo": "name",
        }
    
    def to_json(self) -> str:
        """Export visualization data as JSON."""
        return json.dumps(self.to_plotly_data(), indent=2)
    
    def generate_html(self, include_controls: bool = True) -> str:
        """
        Generate standalone HTML with 3D visualization.
        
        Args:
            include_controls: Whether to include playback controls
            
        Returns:
            HTML string
        """
        plotly_data = self.to_plotly_data()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Packing Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #plot {{ width: 100%; height: 600px; }}
        .controls {{ margin: 20px 0; }}
        button {{ padding: 10px 20px; margin: 5px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>3D Truck Loading Visualization</h1>
    <div id="plot"></div>
    {"<div class='controls'><button onclick='animatePacking()'>Animate Loading</button><button onclick='resetView()'>Reset View</button></div>" if include_controls else ""}
    <script>
        var data = {json.dumps(plotly_data['data'])};
        var layout = {json.dumps(plotly_data['layout'])};
        Plotly.newPlot('plot', data, layout);
        
        function resetView() {{
            Plotly.relayout('plot', {{
                'scene.camera': {{eye: {{x: 1.5, y: 1.5, z: 1.2}}}}
            }});
        }}
        
        function animatePacking() {{
            // Animation logic would go here
            alert('Animation feature - boxes would load one by one');
        }}
    </script>
</body>
</html>
"""
        return html


def create_matplotlib_visualization(container: Dict, boxes: List[Dict]):
    """
    Create matplotlib 3D visualization.
    
    Args:
        container: Container dimensions
        boxes: List of packed boxes
        
    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise ImportError("matplotlib is required for this visualization") from exc
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw container
    L, W, H = container["length"], container["width"], container["height"]
    
    # Container edges
    for s, e in [([0,0,0], [L,0,0]), ([0,0,0], [0,W,0]), ([0,0,0], [0,0,H]),
                 ([L,0,0], [L,W,0]), ([L,0,0], [L,0,H]), ([0,W,0], [L,W,0]),
                 ([0,W,0], [0,W,H]), ([0,0,H], [L,0,H]), ([0,0,H], [0,W,H]),
                 ([L,W,0], [L,W,H]), ([L,0,H], [L,W,H]), ([0,W,H], [L,W,H])]:
        ax.plot3D(*zip(s, e), 'gray', linewidth=1)
    
    # Color palette
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i / max(len(boxes), 1)) for i in range(len(boxes))]
    
    # Draw boxes
    for box, color in zip(boxes, colors):
        pos = box["position"]
        dims = box["dimensions"]
        
        x, y, z = pos["x"], pos["y"], pos["z"]
        dx, dy, dz = dims["length"], dims["width"], dims["height"]
        
        # Create vertices
        vertices = [
            [(x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z)],  # bottom
            [(x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],  # top
            [(x, y, z), (x, y+dy, z), (x, y+dy, z+dz), (x, y, z+dz)],  # left
            [(x+dx, y, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x+dx, y, z+dz)],  # right
            [(x, y, z), (x+dx, y, z), (x+dx, y, z+dz), (x, y, z+dz)],  # front
            [(x, y+dy, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],  # back
        ]
        
        ax.add_collection3d(Poly3DCollection(
            vertices, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5
        ))
    
    ax.set_xlabel('Length (cm)')
    ax.set_ylabel('Width (cm)')
    ax.set_zlabel('Height (cm)')
    ax.set_title('3D Bin Packing Visualization')
    
    # Set equal aspect ratio
    max_dim = max(L, W, H)
    ax.set_xlim([0, max_dim])
    ax.set_ylim([0, max_dim])
    ax.set_zlim([0, max_dim])
    
    return fig
