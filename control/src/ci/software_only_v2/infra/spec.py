"""
spec.py — FleetSpec declarative topology DSL for v2 tests.

A FleetSpec is a builder that accumulates topology declarations (domes, modules,
DAQ nodes, headnode) and produces a frozen Topology on .build(). The Topology
holds validated Pydantic models for all 7 config files plus a NetworkX graph.

Typical usage::

    spec = (
        FleetSpec(seed=42, name="two_dome_test", tier="tier1")
            .with_headnode(ip="10.0.1.5", data_dir="/data/head")
            .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
                .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .build()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Sub-dataclasses (components of the spec, not Pydantic)
# ---------------------------------------------------------------------------

@dataclass
class GatewaySpec:
    """Port-forwarding gateway between head network and DAQ subnet."""
    ip: str
    grpc_port: int = 50051
    ssh_port: int = 22


@dataclass
class _ModuleSpec:
    module_id: int
    version: str          # "qfp" | "bga"
    timing: str           # "wr" | "gnss"
    ip: str               # base IP of the module (quabo 0)
    mobo_serialno: str | None = None
    wps: str | None = None


@dataclass
class _DomeSpec:
    name: str
    lat: float
    lon: float
    alt: float
    modules: list[_ModuleSpec] = field(default_factory=list)


@dataclass
class _DaqNodeSpec:
    ip: str
    module_ids: list[int]
    gateway: GatewaySpec | None = None
    data_dir: str = "/data"
    username: str = "panoseti"
    bindhost: str = "0.0.0.0"


@dataclass
class _DataSpec:
    run_type: str = "engineering"
    overvoltage: int = 2
    integration_time_usec: int = 1000
    pe_threshold: float = 1.0
    quabo_sample_size: int = 16


@dataclass
class _FirmwareSpec:
    qfp: str = "quabo_qfp_stub.bin"
    bga: str = "quabo_bga_stub.bin"


# ---------------------------------------------------------------------------
# Topology (output of FleetSpec.build())
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """
    Frozen snapshot of all 7 validated Pydantic configs for a test fleet.
    Produced by FleetSpec.build().
    """
    # Pydantic models (all validated, ready for .model_dump_json())
    obs: object          # ObsConfig
    daq: object          # DaqConfig
    network: object      # NetworkConfig
    data: object         # DataConfig
    firmware: object     # FirmwareConfig
    quabo_uids: object   # QuaboUids
    daemons: object      # DaemonConfig
    # NetworkX graph (built from daq + quabo_uids + obs + network)
    graph: object        # nx.DiGraph
    name: str = "unnamed"


# ---------------------------------------------------------------------------
# FleetSpec builder
# ---------------------------------------------------------------------------

class FleetSpec:
    """
    Declarative topology builder for v2 test infrastructure.

    Calling .build() validates the topology via GlobalConfigValidator and
    returns a Topology containing all 7 Pydantic config models.

    Convenience factories:
      - FleetSpec.minimal_unit(): single dome, one module, no DAQ node
      - FleetSpec.minimal_fleet(): one dome, one module, one DAQ node (no gateway)

    Setting tier controls which container shape synth.realize() picks:
      - "tier1" / "tier2": no containers; validation only
      - "tier3" / "tier3-lite": sim daqnodes with UdsStrategy
      - "tier4": tier3 + chaos toolkit active
      - "tier5": real hashpipe daqnodes
    """

    def __init__(
        self,
        seed: int = 42,
        name: str = "test_fleet",
        tier: Literal["tier1", "tier2", "tier3", "tier3-lite", "tier4", "tier5"] = "tier1",
    ) -> None:
        self._seed = seed
        self._name = name
        self._tier = tier
        self._headnode_ip: str = "10.0.1.5"
        self._head_data_dir: str = "/data/head"
        self._domes: list[_DomeSpec] = []
        self._daq_nodes: list[_DaqNodeSpec] = []
        self._data_spec = _DataSpec()
        self._firmware_spec = _FirmwareSpec()
        # Track the "current dome" for fluent .add_module() chaining
        self._current_dome: _DomeSpec | None = None

    # ------------------------------------------------------------------
    # Fluent builder methods
    # ------------------------------------------------------------------

    def with_headnode(self, ip: str = "10.0.1.5", data_dir: str = "/data/head") -> "FleetSpec":
        self._headnode_ip = ip
        self._head_data_dir = data_dir
        return self

    def add_dome(
        self,
        name: str,
        lat: float,
        lon: float,
        alt: float,
    ) -> "FleetSpec":
        """Start a new dome context. Subsequent add_module() calls go here."""
        dome = _DomeSpec(name=name, lat=lat, lon=lon, alt=alt)
        self._domes.append(dome)
        self._current_dome = dome
        return self

    def add_module(
        self,
        module_id: int,
        version: str = "qfp",
        timing: str = "wr",
        ip: str = "",
        mobo_serialno: str | None = None,
        wps: str | None = None,
    ) -> "FleetSpec":
        """Add a module to the current dome context."""
        if self._current_dome is None:
            raise RuntimeError("Call add_dome() before add_module()")
        if not ip:
            raise ValueError(f"Module {module_id} requires an explicit ip address")
        mod = _ModuleSpec(
            module_id=module_id,
            version=version,
            timing=timing,
            ip=ip,
            mobo_serialno=mobo_serialno or f"M{module_id:03d}",
            wps=wps,
        )
        self._current_dome.modules.append(mod)
        return self

    def add_daq_node(
        self,
        ip: str,
        modules: list[int],
        gateway: GatewaySpec | None = None,
        data_dir: str = "/data",
        username: str = "panoseti",
        bindhost: str = "0.0.0.0",
    ) -> "FleetSpec":
        self._daq_nodes.append(_DaqNodeSpec(
            ip=ip,
            module_ids=modules,
            gateway=gateway,
            data_dir=data_dir,
            username=username,
            bindhost=bindhost,
        ))
        return self

    def with_data(
        self,
        run_type: str = "engineering",
        overvoltage: int = 2,
        integration_time_usec: int = 1000,
        pe_threshold: float = 1.0,
        quabo_sample_size: int = 16,
    ) -> "FleetSpec":
        self._data_spec = _DataSpec(
            run_type=run_type,
            overvoltage=overvoltage,
            integration_time_usec=integration_time_usec,
            pe_threshold=pe_threshold,
            quabo_sample_size=quabo_sample_size,
        )
        return self

    def with_firmware(self, qfp: str = "quabo_qfp_stub.bin", bga: str = "quabo_bga_stub.bin") -> "FleetSpec":
        self._firmware_spec = _FirmwareSpec(qfp=qfp, bga=bga)
        return self

    def build(self) -> Topology:
        """Synthesize and validate configs, returning a Topology."""
        from ci.software_only_v2.infra.synth import realize
        return realize(self)

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def minimal_unit(cls) -> "FleetSpec":
        """Single dome + one module + one DAQ node. Minimal topology that passes all validators."""
        return (
            cls(seed=0, name="minimal_unit", tier="tier1")
            .with_headnode(ip="10.0.1.5", data_dir="/data/head")
            .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .add_daq_node(ip="192.168.0.10", modules=[200], bindhost="lo")
        )

    @classmethod
    def minimal_fleet(cls) -> "FleetSpec":
        """Single dome + one module + one DAQ node. Smallest valid fleet."""
        return (
            cls(seed=1, name="minimal_fleet", tier="tier3")
            .with_headnode(ip="10.0.1.5", data_dir="/data/head")
            .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .add_daq_node(ip="192.168.0.10", modules=[200], bindhost="lo")
        )

    @classmethod
    def two_node_ci(
        cls,
        head_prefix: str = "10.0.1",
        daq_prefix: str = "192.168.0",
        quabo_prefix: str = "192.168.3",
        tier: str = "tier3",
    ) -> "FleetSpec":
        """Two-node CI fleet matching the static compose topology."""
        from control.utils.config_file import ip_addr_to_module_id
        mod1_ip = f"{quabo_prefix}.32"
        mod2_ip = f"{quabo_prefix}.36"
        mid1 = ip_addr_to_module_id(mod1_ip)
        mid2 = ip_addr_to_module_id(mod2_ip)
        return (
            cls(seed=2, name="two_node_ci", tier=tier)  # type: ignore[arg-type]
            .with_headnode(ip=f"{head_prefix}.22", data_dir="/data/head")
            .add_dome("ci_dome", lat=37.0, lon=-121.0, alt=1000.0)
            .add_module(mid1, version="qfp", timing="wr", ip=mod1_ip)
            .add_module(mid2, version="qfp", timing="wr", ip=mod2_ip)
            .add_daq_node(ip=f"{daq_prefix}.10", modules=[mid1], bindhost="lo")
            .add_daq_node(ip=f"{daq_prefix}.20", modules=[mid2], bindhost="lo")
        )

    # ------------------------------------------------------------------
    # Internal accessors (used by synth.py)
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def headnode_ip(self) -> str:
        return self._headnode_ip

    @property
    def head_data_dir(self) -> str:
        return self._head_data_dir

    @property
    def domes(self) -> list[_DomeSpec]:
        return self._domes

    @property
    def daq_nodes(self) -> list[_DaqNodeSpec]:
        return self._daq_nodes

    @property
    def data_spec(self) -> _DataSpec:
        return self._data_spec

    @property
    def firmware_spec(self) -> _FirmwareSpec:
        return self._firmware_spec
