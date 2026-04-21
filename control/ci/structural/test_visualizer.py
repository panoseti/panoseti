"""
test_visualizer.py

Unit tests for the topology visualizer.
"""

from __future__ import annotations

import pathlib
import tempfile

import networkx as nx

from control.topology.visualizer import export_topology_json, save_topology_image


def test_save_topology_image():
    """Verify that save_topology_image creates a file."""
    graph = nx.DiGraph()
    graph.add_node("head", role="headnode", label="Head")
    graph.add_node("daq", role="daqnode", label="DAQ")
    graph.add_edge("head", "daq", type="control")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = pathlib.Path(tmp_dir) / "test.png"
        save_topology_image(graph, img_path)
        assert img_path.exists(), "PNG image not created"
        assert img_path.stat().st_size > 0, "PNG image is empty"


def test_export_topology_json():
    """Verify that export_topology_json creates a valid JSON file."""
    graph = nx.DiGraph()
    graph.add_node("head", role="headnode", label="Head")
    graph.add_edge("head", "daq", type="control")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = pathlib.Path(tmp_dir) / "test.json"
        export_topology_json(graph, json_path)
        assert json_path.exists(), "JSON file not created"
        
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 1
