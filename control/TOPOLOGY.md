# PANOSETI Topology Management

This directory contains utilities for modeling, visualizing, and validating the PANOSETI DAQ fleet topology using [NetworkX](https://networkx.org/).

## 🚀 Overview

The topology system allows operators and developers to:
1.  **Model** the fleet as a directed graph (DiGraph).
2.  **Visualize** network and hardware relationships (Head Node → DAQ Node → Gateway → Quabo).
3.  **Validate** structural integrity (detecting orphans, bottlenecks, and control loops) before touching hardware.

## 🛠️ Components

### 1. GraphBuilder (`src/control/topology/graph_builder.py`)
The core engine that parses Pydantic configuration models and constructs a NetworkX graph.
- **Nodes:** HeadNode, DAQNode, Gateway, Module, Quabo.
- **Edges:** Control (gRPC), Network (SSH Tunnel), Data (UDP), Logical (Aggregation).

### 2. Visualizer (`src/control/topology/visualizer.py`)
Tools to render the graph to images or export to standard JSON formats.
- `save_topology_image(graph, path)`: Generates a PNG using Matplotlib.
- `export_topology_json(graph, path)`: Exports to Cytoscape/D3.js compatible format.

### 3. Fleet Generator (`src/control/topology/fleet.py`)
Utility to programmatically create complex $n$-node configurations for testing and simulation.

## 📈 Usage Examples

### Building and Saving a Graph
```python
from control.topology.graph_builder import GraphBuilder
from control.topology.visualizer import save_topology_image
from control.utils import config_file

# 1. Load configs
daq_config = config_file.get_daq_config()
quabo_uids = config_file.get_quabo_uids()

# 2. Build graph
builder = GraphBuilder()
graph = builder.build_from_configs(daq_config, quabo_uids)

# 3. Save as PNG
save_topology_image(graph, "topology.png")
```

### Generating an n-node Fleet
```python
from control.topology.fleet import generate_fleet_configs

# Create a 10-node fleet with 2 modules per node and 50% subnet probability
daq_config, quabo_uids = generate_fleet_configs(
    num_daq_nodes=10, 
    modules_per_node=2,
    subnet_probability=0.5
)
```

## 🧪 Structural Simulations

The parameterized simulation suite (`ci/structural/test_observatory_simulation.py`) allows testing against complex, randomized topologies. These tests verify:
- **Graph Completeness:** Every logical node (Quabo) has a physical parent.
- **Reachability:** The Head Node has a valid command propagation path to every DAQ node and Quabo.
- **Subnet Integrity:** Nodes behind gateways maintain their downstream hardware links.

## 🛡️ Structural Validation

The `GlobalConfigValidator` (in `src/control/utils/global_validator.py`) now includes graph-based checks:

| Check | Logic | Severity |
| :--- | :--- | :--- |
| **Topology Reachability** | Ensures all Quabos/Modules are reachable from the Head Node. | **ERROR** |
| **DAQ Node Bottleneck** | Warns if a DAQ node manages more than `daq_node_module_limit` (default: 4). | **WARN** |
| **Control Loop Check** | Ensures the topology is a Directed Acyclic Graph (DAG). | **ERROR** |

### Configuring Bottlenecks
You can adjust the module limit in your `daq_config.json`:
```json
{
    "head_node_ip_addr": "10.0.1.5",
    "daq_node_module_limit": 8,
    "daq_nodes": [...]
}
```

## 🧪 Testing

Run the high-speed structural test suite:
```bash
uv run pytest ci/structural/
```
