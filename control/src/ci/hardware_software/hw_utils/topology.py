"""
Hardware Topology adapter.
Loads obs/daq/network configs and derives hardware capabilities for test gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QuaboAddr:
    ip: str
    module_id: int
    quadrant: int
    boardloc: int


@dataclass
class DaqNode:
    host: str
    module_ids: list[int]


@dataclass
class WpsOutlet:
    name: str
    url: str
    quabo_socket: int


class HwTopology:
    """
    Reads the active observatory configs and exposes hardware topology.

    All topology data is derived from the three canonical config files —
    no IPs or module IDs are hardcoded.
    """

    def __init__(self, config_dir: Path | str | None = None):
        from control.utils import config_file  # lazy to avoid import at collection time
        self._obs = config_file.get_obs_config()
        self._daq = config_file.get_daq_config()
        self._net = config_file.get_network_config()

    # ── Public accessors ──────────────────────────────────────────────────────

    def quabo_ips(self) -> list[QuaboAddr]:
        """All quabos in the active observatory layout."""
        from control.utils.config_file import get_boardloc, ip_addr_to_module_id
        result: list[QuaboAddr] = []
        for dome in self._obs.domes:
            for module in dome.modules:
                base_ip: str = str(module.ip_addr)
                parts = base_ip.split(".")
                for q in range(4):
                    ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{int(parts[3]) + q}"
                    mid = ip_addr_to_module_id(base_ip)
                    result.append(QuaboAddr(
                        ip=ip,
                        module_id=mid,
                        quadrant=q,
                        boardloc=get_boardloc(base_ip, q),
                    ))
        return result

    def daq_nodes(self) -> list[DaqNode]:
        """All DAQ nodes in the active config."""
        result: list[DaqNode] = []
        for node in self._daq.daq_nodes:
            result.append(DaqNode(
                host=str(node.ip_addr),
                module_ids=node.module_ids,
            ))
        return result

    def wps_outlets(self) -> list[WpsOutlet]:
        """All WPS outlets defined in obs_config."""
        result: list[WpsOutlet] = []
        extra_data = self._obs.model_extra or {}
        for key, val in extra_data.items():
            if key.startswith("wps") and isinstance(val, dict):
                result.append(WpsOutlet(
                    name=key,
                    url=val.get("url", ""),
                    quabo_socket=val.get("quabo_socket", 1),
                ))
        return result

    def module_ids(self) -> list[int]:
        """Unique module IDs across all quabos."""
        return sorted({q.module_id for q in self.quabo_ips()})

    def capabilities(self) -> set[str]:
        """Derive hardware capability flags from the active configs."""
        caps: set[str] = set()

        # White Rabbit: any module with timing_mode == "wr"
        for dome in self._obs.domes:
            for module in dome.modules:
                if module.timing_mode == "wr":
                    caps.add("white_rabbit")

        # GNSS: any module with timing_mode == "gnss"
        for dome in self._obs.domes:
            for module in dome.modules:
                if module.timing_mode == "gnss":
                    caps.add("gnss")

        # Port forwarding: non-empty network_config port_forwarding list
        has_pf = any(n.port_forwarding and n.port_forwarding.status for n in self._net.daq_nodes) or \
                 any(m.port_forwarding and m.port_forwarding.status for m in self._net.modules)
        if has_pf:
            caps.add("port_forwarding")

        # Multi-module: more than one module
        if len(self.module_ids()) > 1:
            caps.add("multi_module")

        return caps

    def gate(self, test_id: str, requirement: dict[str, Any]) -> bool | str:
        """
        Check whether the active topology meets a requirement entry.

        Returns True if the test may run, or a string skip reason if not.
        """
        caps = self.capabilities()

        cap = requirement.get("requires_capability")
        if cap and cap not in caps:
            return f"Topology lacks capability '{cap}' (active: {sorted(caps)})"

        min_modules = requirement.get("requires_min_modules")
        if min_modules and len(self.module_ids()) < min_modules:
            return f"Need ≥{min_modules} modules, found {len(self.module_ids())}"

        min_daq = requirement.get("requires_min_daq_nodes")
        if min_daq and len(self.daq_nodes()) < min_daq:
            return f"Need ≥{min_daq} DAQ nodes, found {len(self.daq_nodes())}"

        return True
