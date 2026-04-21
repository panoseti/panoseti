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
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Define Layout
    # Use spring_layout for general graph, but we can attempt a layered one
    pos = nx.spring_layout(graph, k=0.5, iterations=50)

    # 2. Node Styling
    node_colors = []
    node_sizes = []
    for _node, data in graph.nodes(data=True):
        role = data.get("role", "unknown")
        if role == "headnode":
            node_colors.append("red")
            node_sizes.append(1000)
        elif role == "daqnode":
            node_colors.append("royalblue")
            node_sizes.append(800)
        elif role == "gateway":
            node_colors.append("forestgreen")
            node_sizes.append(800)
        elif role == "module":
            node_colors.append("orange")
            node_sizes.append(500)
        elif role == "quabo":
            node_colors.append("skyblue")
            node_sizes.append(200)
        else:
            node_colors.append("gray")
            node_sizes.append(300)

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
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    # Draw edges one type at a time to support styles
    for style in set(edge_styles):
        typed_edges = [
            (u, v)
            for (u, v, _d), s in zip(graph.edges(data=True), edge_styles, strict=True)
            if s == style
        ]
        nx.draw_networkx_edges(
            graph, pos, edgelist=typed_edges, style=style, arrows=True, width=1.5
        )


    labels = {node: data.get("label", str(node)) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_family="sans-serif")

    plt.title("PANOSETI Hardware & Network Topology")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def export_topology_json(graph: nx.DiGraph, path: str | pathlib.Path) -> None:
    """
    Exports the graph to a JSON format compatible with Cytoscape.js or D3.js.
    """
    data = nx.node_link_data(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
