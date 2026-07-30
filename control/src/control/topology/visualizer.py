"""
visualizer.py

Visualization tools for the PANOSETI topology graph.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import networkx as nx


def save_topology_image(graph: nx.DiGraph, path: str | pathlib.Path) -> None:
    """
    Renders the topology graph to an image file.
    Uses edge weights to pull Quabos close to their modules, creating a hub-spoke look.
    """
    plt.figure(figsize=(16, 12))
    
    # 1. Define Layout
    # Assign higher weights to logical edges (Module -> Quabo) to keep them close
    for u, v, d in graph.edges(data=True):
        if d.get("type") == "logical":
            graph[u][v]["weight"] = 10.0
        else:
            graph[u][v]["weight"] = 1.0

    # Initialize with multipartite to keep tiers separate
    init_pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
    
    # Run spring layout with weights
    pos = nx.spring_layout(graph, pos=init_pos, k=0.15, iterations=100, weight="weight")

    # 2. Node Styling
    node_colors = []
    node_sizes = []
    for _node, data in graph.nodes(data=True):
        role = data.get("role", "unknown")
        if role == "headnode":
            node_colors.append("#d32f2f") # Red
            node_sizes.append(1500)
        elif role == "daqnode":
            node_colors.append("#1976d2") # Blue
            node_sizes.append(1200)
        elif role == "gateway":
            node_colors.append("#388e3c") # Green
            node_sizes.append(1200)
        elif role == "module":
            node_colors.append("#f57c00") # Orange
            node_sizes.append(800)
        elif role == "quabo":
            node_colors.append("#03a9f4") # Sky Blue
            node_sizes.append(400)
        else:
            node_colors.append("#9e9e9e") # Gray
            node_sizes.append(500)

    # 3. Edge Styling
    edge_styles = []
    for _u, _v, data in graph.edges(data=True):
        e_type = data.get("type", "unknown")
        if e_type == "control":
            edge_styles.append("solid")
        elif e_type == "data":
            edge_styles.append("dashed")
        elif e_type == "network":
            edge_styles.append("solid")
        elif e_type == "logical":
            edge_styles.append("dotted")
        else:
            edge_styles.append("solid")

    # 4. Draw
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors="black")
    
    # Draw edges one type at a time to support styles
    for style in set(edge_styles):
        typed_edges = [
            (u, v)
            for (u, v, _d), s in zip(graph.edges(data=True), edge_styles, strict=True)
            if s == style
        ]
        nx.draw_networkx_edges(
            graph, pos, edgelist=typed_edges, style=style, arrows=True, width=1.8, edge_color="#333333"
        )

    labels = {node: data.get("label", str(node)) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_family="sans-serif", font_weight="bold")

    plt.title("PANOSETI Hub-Spoke Fleet Topology", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def export_topology_json(graph: nx.DiGraph, path: str | pathlib.Path) -> None:
    """
    Exports the graph to a JSON format compatible with Cytoscape.js or D3.js.
    """
    data = nx.node_link_data(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def export_interactive_html(graph: nx.DiGraph, path: str | pathlib.Path) -> None:
    """
    Creates an interactive HTML visualization using Pyvis.
    """
    from pyvis.network import Network

    # Use notebook=False to ensure valid standalone HTML generation
    net = Network(notebook=False, directed=True, height="800px", width="100%", bgcolor="#ffffff", font_color="#000000")
    
    # Configure physics for a balanced hub-spoke structure
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )

    colors = {
        "headnode": "#d32f2f", 
        "daqnode": "#1976d2", 
        "gateway": "#388e3c", 
        "module": "#f57c00", 
        "quabo": "#03a9f4"
    }

    for node, data in graph.nodes(data=True):
        role = data.get("role", "unknown")
        label = data.get("label", str(node)).replace("\n", " ")
        net.add_node(
            node, 
            label=label, 
            title=f"Role: {role}<br>IP: {data.get('ip')}", 
            color=colors.get(role, "#9e9e9e"),
            size=30 if role != "quabo" else 20
        )

    for u, v, data in graph.edges(data=True):
        e_type = data.get("type", "unknown")
        label = data.get("label", "")
        # dashed edges for data
        dashed = (e_type == "data" or e_type == "logical")
        arrows = 'to;from' if (e_type != 'data') else 'from'
        net.add_edge(u, v, title=e_type, label=label, dashed=dashed, arrows=arrows)

    net.write_html(str(path))
