"""
global_validator.py

Executes Tier-2 cross-configuration validations to ensure physical, network,
and hardware states are cohesive across the entire PanoSETI observatory.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from typing import Any, TypeVar, cast

from haversine import Unit, haversine  # type: ignore[import-untyped]
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from control.utils import util
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    FirmwareConfig,
    NetworkConfig,
    ObsConfig,
    QuaboUids,
)

console = Console()

MAX_DOME_BASELINE_KM = 2
MAX_MODULES_PER_DAQ_NODE = 3

class ValidationReport:
    """Aggregates tests for a unified pre-flight report."""

    def __init__(self) -> None:
        self.tests: list[dict[str, str]] = []
        self.has_errors = False

    def add_test(self, name: str, status: str, info: str = "") -> None:
        self.tests.append({"name": name, "status": status, "info": info})
        if status == "ERROR":
            self.has_errors = True

    def print_report(self) -> None:
        table = Table(title="Global Tier-2 Validation Report", show_lines=True)
        table.add_column("Test Name", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for t in self.tests:
            if t["status"] == "PASS":
                status_fmt = "[green]PASS[/green]"
            elif t["status"] == "WARN":
                status_fmt = "[yellow]WARN[/yellow]"
            else:
                status_fmt = "[red]ERROR[/red]"

            table.add_row(t["name"], status_fmt, t["info"])

        console.print(table)


T = TypeVar("T", bound=BaseModel)

def validate_all(
    obs_config: dict[str, Any] | ObsConfig,
    data_config: dict[str, Any] | DataConfig,
    daq_config: dict[str, Any] | DaqConfig | None = None,
    network_config: dict[str, Any] | NetworkConfig | None = None,
    firmware_config: dict[str, Any] | FirmwareConfig | None = None,
    quabo_uids: dict[str, Any] | QuaboUids | None = None,
) -> bool:
    """Unified entry point for global validation."""
    from control.utils.pydantic_config_models import (
        DaqConfig,
        DataConfig,
        FirmwareConfig,
        NetworkConfig,
        ObsConfig,
    )

    def _ensure_model(cfg: dict[str, Any] | T | None, model: type[T]) -> T | None:
        if cfg is None:
            return None
        if isinstance(cfg, model):
            return cfg
        return model(**cast(dict[str, Any], cfg))

    validated_configs: dict[str, BaseModel | None] = {
        'obs': _ensure_model(obs_config, ObsConfig),
        'data': _ensure_model(data_config, DataConfig),
        'daq': _ensure_model(daq_config, DaqConfig),
        'network': _ensure_model(network_config, NetworkConfig),
        'firmware': _ensure_model(firmware_config, FirmwareConfig),
        'uids': _ensure_model(quabo_uids, QuaboUids),
    }

    validator = GlobalConfigValidator(validated_configs)
    success = validator.validate_all_rules()
    if not success:
        errors = [f"{t['name']}: {t['info']}" for t in validator.report.tests if t['status'] == 'ERROR']
        raise ValueError(f"Global configuration validation failed: {'; '.join(errors)}")
    return success


class GlobalConfigValidator:
    """Executes cross-configuration validation rules for the observatory.
    
    This validator ensures that individual configuration files (obs, data, 
    daq, network, firmware) are mutually consistent and physically plausible.
    """

    def __init__(self, validated_configs: dict[str, BaseModel | None]):
        """Initialize the validator with validated configuration models.

        Args:
            validated_configs: Dictionary containing Pydantic model instances 
                               for each configuration type.
        """
        self.obs_conf = cast(ObsConfig, validated_configs.get('obs'))
        self.data_conf = cast(DataConfig, validated_configs.get('data'))
        self.daq_conf = cast(DaqConfig, validated_configs.get('daq'))
        self.net_conf = cast(NetworkConfig, validated_configs.get('network'))
        self.firmware_conf = cast(FirmwareConfig, validated_configs.get('firmware'))
        self.uids = cast(QuaboUids, validated_configs.get('uids'))
        self.report = ValidationReport()
        util.attach_daq_config(self.daq_conf, self.net_conf)

    def validate_all_rules(self) -> bool:
        """Execute all validation methods prefixed with '_check_'.

        Returns:
            True if all rules passed (including warnings), False if any ERROR occurred.
        """
        rule_methods = [getattr(self, func) for func in dir(self) if
                        callable(getattr(self, func)) and func.startswith("_check_")]
        for rule in rule_methods:
            with contextlib.suppress(ValueError):
                rule()
        self.report.print_report()
        return not self.report.has_errors

    def _check_science_guardrails(self) -> None:
        """Warn if flash or stim signals are enabled for a science run.

        Prevents accidental injection of artificial signals into real science data.
        """
        if not self.data_conf:
            return
        run_type = self.data_conf.run_type.lower()
        if "eng" not in run_type:
            flash_on = self.data_conf.flash_params is not None
            stim_on = self.data_conf.stim_params is not None
            if flash_on or stim_on:
                self.report.add_test("Science Guardrails", "WARN",
                                     f"Run type '{run_type}' has flash/stim enabled. Artificial signals will be injected.")
                return
        self.report.add_test("Science Guardrails", "PASS", f"Run type: '{run_type}'")

    def _check_geospatial_coherence(self) -> None:
        """Ensure all domes are within a reasonable physical distance (2km).

        Detects decimal point errors or incorrect observatory coordinates.
        """
        if not self.obs_conf:
            return
        domes = self.obs_conf.domes
        if len(domes) < 2:
            self.report.add_test("Geospatial Coherence", "PASS", "Only one dome defined.")
            return

        coords = [(d.name, d.obslat, d.obslon) for d in domes]
        max_dist = 0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = haversine((coords[i][1], coords[i][2]), (coords[j][1], coords[j][2]), unit=Unit.KILOMETERS)
                max_dist = max(max_dist, dist)

        if max_dist > MAX_DOME_BASELINE_KM:
            self.report.add_test("Geospatial Coherence", "ERROR",
                                 f"Domes are {max_dist:.2f} km apart (> {MAX_DOME_BASELINE_KM} km max baseline). Check decimal places.")
        else:
            self.report.add_test("Geospatial Coherence", "PASS", f"Max baseline: {max_dist:.2f} km")

    def _check_network_tunneling(self) -> None:
        """Verify that all defined modules have corresponding network routing.

        Warns if a module is defined in obs_config but lacks a port forwarding 
        entry in network_config when using a non-local network.
        """
        if not self.obs_conf or not self.net_conf:
            return
        obs_ips = {str(m.ip_addr) for d in self.obs_conf.domes for m in d.modules}
        net_mapped_ips = {str(m.ip_addr) for m in self.net_conf.modules if
                          m.port_forwarding.status}

        missing = obs_ips - net_mapped_ips
        if missing and self.net_conf.modules:
            self.report.add_test("Network Tunneling Mapping", "WARN",
                                 f"Modules lacking port forwarding: {', '.join(missing)}")
        else:
            self.report.add_test("Network Tunneling Mapping", "PASS", "All modules accounted for in routing.")

    def _check_hardware_firmware(self) -> None:
        """Verify that firmware binaries are configured for all active hardware types (BGA/QFP)."""
        if not self.obs_conf or not self.firmware_conf:
            return
        required_hw = set()
        for d in self.obs_conf.domes:
            for m in d.modules:
                qv = m.quabo_version
                if isinstance(qv, list):
                    required_hw.update(qv)
                else:
                    required_hw.add(qv)

        # Firmware model has explicit fields: qfp, bga, gold
        firmware_keys = set()
        if self.firmware_conf.qfp:
            firmware_keys.add("qfp")
        if self.firmware_conf.bga:
            firmware_keys.add("bga")
        if self.firmware_conf.gold:
            firmware_keys.add("gold")

        # Plus any dynamic keys in model_extra
        extra = self.firmware_conf.model_extra or {}
        firmware_keys.update(extra.keys())

        # If 'quabo' and 'wr' are top-level keys, look inside them
        # (Based on test_sc_config_validation.py's mock structure)
        if "quabo" in extra:
            firmware_keys.update(extra["quabo"].keys())

        missing = required_hw - firmware_keys
        if missing:
            self.report.add_test("Hardware-Firmware Alignment", "ERROR", f"Missing binary configurations for HW types: {missing}. {self.firmware_conf=}")
        else:
            self.report.add_test("Hardware-Firmware Alignment", "PASS", "Binary configurations exist for all active hardware.")
    def _check_firmware_filesystem(self) -> None:
        """Strictly verify that configured firmware files actually exist on disk."""
        if not self.firmware_conf:
            return
        
        from pathlib import Path
        extra = self.firmware_conf.model_extra or {}
        checks_performed = 0
        
        # 1. Check WR firmware path
        wr_conf = extra.get("wr", {})
        if wr_conf and "wrpc_filesys" in wr_conf:
            path = Path(wr_conf["wrpc_filesys"])
            if not path.exists():
                self.report.add_test("Firmware Path Check", "ERROR", 
                                     f"WR firmware path '{path}' does not exist on this host.")
            else:
                self.report.add_test("Firmware Path Check", "PASS", "WR firmware path verified.")
            checks_performed += 1

        # 2. Check Quabo binary files
        # Support both {"quabo": {"bga": "..."}} and {"bga": "..."}
        quabo_conf = extra.get("quabo", {})
        if not quabo_conf:
            # Fallback: assume all top-level keys except 'wr' are quabo hardware types
            quabo_conf = {k: v for k, v in extra.items() if k != "wr" and isinstance(v, str)}

        for hw_type, binary_path in quabo_conf.items():
            path = Path(binary_path)
            if not path.is_file() and not (Path(".") / path).is_file():
                self.report.add_test("Firmware Binary Check", "ERROR", 
                                     f"Quabo binary file '{binary_path}' (type: {hw_type}) does not exist.")
            else:
                self.report.add_test("Firmware Binary Check", "PASS", f"Binary for {hw_type} verified.")
            checks_performed += 1
        
        if checks_performed == 0:
            self.report.add_test("Firmware Filesystem", "PASS", "No firmware files configured for disk check.")

    def _check_overvoltage_consensus(self) -> None:
        """Ensure detector overvoltage is consistent across obs and data configs."""
        if not self.obs_conf or not self.data_conf:
            return
        obs_ov = self.obs_conf.detector_overvoltage
        data_ov = self.data_conf.detector_overvoltage
        if obs_ov is not None and data_ov is not None and obs_ov != data_ov:
            self.report.add_test("Overvoltage Consensus", "ERROR",
                                 f"obs_config ({obs_ov}V) != data_config ({data_ov}V)")
        else:
            self.report.add_test("Overvoltage Consensus", "PASS", f"Voltage aligned at {obs_ov}V")

    def _check_port_collisions(self) -> None:
        """Ensure multiple modules on a Gateway do not use overlapping forwarded ports."""
        if not self.net_conf:
            return
        gw_ports: dict[str, set[int]] = {}
        for m in self.net_conf.modules:
            pf = m.port_forwarding
            if pf.status:
                gw = str(pf.gw_ip)
                ports = (pf.cmd_port or []) + (pf.reboot_port or [])
                if gw not in gw_ports:
                    gw_ports[gw] = set()
                for p in ports:
                    if p is None:
                        continue
                    if p in gw_ports[gw]:
                        self.report.add_test("Port Collision", "ERROR",
                                             f"Gateway {gw} has multiple modules attempting to forward port {p}.")
                        return
                    gw_ports[gw].add(p)
        self.report.add_test("Port Collision", "PASS", "No forwarded port overlaps detected on gateways.")

    def _check_timing_port_collisions(self) -> None:
        """Ensure WR switches and GNSS modules do not share IPs (UDP port contention)."""
        if not self.obs_conf:
            return
        
        wr_ips = set()
        if self.obs_conf.wr_ip_addr:
            wr_ips.add(str(self.obs_conf.wr_ip_addr))
            
        for dome in self.obs_conf.domes:
            for module in dome.modules:
                if module.timing_mode == 'gnss':
                    # Check if GNSS module IP or its explicit wr_ip_addr collides with a WR switch
                    # (In some configs, gnss modules might have a wr_ip_addr field used for other purposes)
                    m_ip = str(module.ip_addr)
                    m_wr_ip = str(module.wr_ip_addr) if hasattr(module, 'wr_ip_addr') and module.wr_ip_addr else None
                    
                    if m_ip in wr_ips or (m_wr_ip and m_wr_ip in wr_ips):
                        self.report.add_test("Timing Port Collision", "ERROR",
                                             f"Module {m_ip} (GNSS) collides with WR switch IP {wr_ips}. "
                                             "Contention for UDP timing ports detected.")
                        return
        self.report.add_test("Timing Port Collision", "PASS", "No timing IP collisions detected.")

    def _check_ph_baselines(self) -> None:
        """Check if Pulse Height baselines are within valid bounds (600-800)."""
        from control.utils import config_file
        try:
            # Load and validate baselines
            baselines = config_file.get_quabo_ph_baselines()
            valid = config_file.validate_ph_baselines(baselines)
            if not valid:
                 self.report.add_test("PH Baseline Calibration", "WARN", "Some baseline values are out of range (600-800).")
            else:
                 self.report.add_test("PH Baseline Calibration", "PASS", "All baseline values within range.")
        except FileNotFoundError:
             self.report.add_test("PH Baseline Calibration", "WARN", "Baseline file missing. Run: pseti config calibrate-ph")
        except Exception as e:
             self.report.add_test("PH Baseline Calibration", "ERROR", f"Validation error: {e}")

    # --- NEW TEST 2: DAQ Assignment Overlap Check ---
    def _check_module_id_uniqueness(self) -> None:
        """Ensure every module (physical board) has a unique derived module_id."""
        if not self.obs_conf:
            return
        from control.utils import config_file
        seen_mids: dict[int, str] = {}
        for dome in self.obs_conf.domes:
            for module in dome.modules:
                mid = config_file.ip_addr_to_module_id(str(module.ip_addr))
                if mid in seen_mids:
                    self.report.add_test("Module ID Uniqueness", "ERROR",
                                         f"Module ID {mid} (from {module.ip_addr}) is already assigned in dome '{seen_mids[mid]}'. "
                                         "This will cause a BOARDLOC collision.")
                    return
                seen_mids[mid] = dome.name
        self.report.add_test("Module ID Uniqueness", "PASS", f"All {len(seen_mids)} modules have unique IDs.")

    def _check_daq_assignment_overlap(self) -> None:
        """Ensure a single module ID is not handled by multiple DAQ nodes."""
        if not self.daq_conf:
            return
        from control.utils.config_file import expand_ranges
        seen_ids: set[int] = set()
        expand_ranges(self.daq_conf)
        for daq in self.daq_conf.daq_nodes:
            ids = daq.module_ids
            overlap = seen_ids.intersection(ids)
            if overlap:
                self.report.add_test("DAQ Overlap", "ERROR",
                                     f"Module IDs {overlap} are assigned to multiple DAQ nodes!")
                return
            seen_ids.update(ids)
        self.report.add_test("DAQ Overlap", "PASS", "No overlapping module_ids across DAQ nodes.")

    def _estimate_data_usage(self) -> tuple[float, float, str]:
        """Estimate the data generation rate and total volume for an 8-hour run.

        Returns:
            A tuple of (TB_per_hour, total_estimated_TB, formula_string).
        """
        if not self.obs_conf or not self.data_conf:
            return 0.0, 0.0, "N/A"
        num_modules = sum(len(d.modules) for d in self.obs_conf.domes)
        img_conf = self.data_conf.image
        if num_modules == 0 or not img_conf:
            return 0.0, 0.0, "N/A"

        int_usec = img_conf.integration_time_usec
        # nsum is not in ImageMode in pydantic_config_models.py, using 1
        nsum = 1 
        bytes_pp = img_conf.quabo_sample_size / 8

        fps = 1_000_000 / (int_usec * nsum)
        tb_per_hr = (fps * 4 * 1024 * bytes_pp * 3600 * num_modules) / (1024 ** 4)
        total_tb = tb_per_hr * 8  # Assume 8 hour run

        formula = f"({fps:.1f}frame/sec * 4quabo/mod * 1024px/quabo * {bytes_pp}B/px * 3600sec * {num_modules}mod) / (1024^4 B/TB)"
        return tb_per_hr, total_tb, formula

    def _check_headnode_disk_space(self) -> None:
        """Verify that the head node has sufficient disk space for the estimated run."""
        if not self.daq_conf:
            return
        head_dir = self.daq_conf.head_node_data_dir
        is_container = getattr(self.daq_conf, "head_node_container", False)
        
        if not head_dir or not os.path.exists(head_dir):
            status = "WARN" if is_container else "ERROR"
            self.report.add_test("Headnode Disk Space", status, f"Path '{head_dir}' missing or unreachable.")
            return

        _total, _used, free = shutil.disk_usage(head_dir)
        free_tb = free / (1024 ** 4)

        tb_per_hr, est_total, formula = self._estimate_data_usage()

        msg = f"Available: {free_tb:.2f} TB. Est: {tb_per_hr:.3f} TB/hr. Formula: {formula}"
        
        # In CI/Docker environments, we might be running on a small disk partition.
        # If head_node_container is True, downgrade ERROR to WARN to allow tests to proceed.
        is_container = getattr(self.daq_conf, "head_node_container", False)
        
        if est_total > 0 and (free_tb - est_total) <= 0:
            status = "WARN" if is_container else "ERROR"
            prefix = "Low Space (CI/Container Mode):" if is_container else "INSUFFICIENT SPACE!"
            self.report.add_test("Headnode Disk Space", status, f"{prefix} {msg}")
        elif free_tb < 1.0:
            self.report.add_test("Headnode Disk Space", "WARN", f"Low Space! {msg}")
        else:
            self.report.add_test("Headnode Disk Space", "PASS", msg)

    def _check_wps_references(self) -> None:
        """Ensure all Web Power Switches referenced by modules are defined in obs_config."""
        if not self.obs_conf:
            return

        missing_wps: set[str] = set()
        wps_keys = set((self.obs_conf.model_extra or {}).keys())
        for d in self.obs_conf.domes:
            for m in d.modules:
                # The default is 'wps' if not specified
                wps_name = m.wps or 'wps'
                if wps_name not in wps_keys:
                    missing_wps.add(wps_name)

        if missing_wps:
            self.report.add_test("WPS Reference Map", "ERROR",
                                 f"Modules reference undefined WPS units: {', '.join(missing_wps)}")
        else:
            self.report.add_test("WPS Reference Map", "PASS", "All referenced WPS units exist in obs_config.")

    def _check_topology_structural_integrity(self) -> None:
        """Use GraphBuilder and NetworkX to find logical flaws in the fleet topology."""
        if not self.daq_conf or not self.obs_conf:
            return
            
        import networkx as nx

        from control.topology.graph_builder import GraphBuilder

        # 1. Build the graph
        from control.utils import config_file
        quabo_uids = self.uids or config_file.get_quabo_uids(exit_on_missing=False)
        if not quabo_uids:
             self.report.add_test("Topology Reachability", "WARN", "quabo_uids.json missing; skipping full reachability check.")
             return
             
        try:
            config_file.associate(self.daq_conf, quabo_uids)
        except ValueError as e:
            # raise ValueError from e

            self.report.add_test("Topology Reachability", "ERROR", f"{e!r}")
            return
            
        
        builder = GraphBuilder()
        graph = builder.build_from_configs(self.daq_conf, quabo_uids, self.obs_conf, self.net_conf)
        
        # 2. Check for Orphans (unreachable from headnode)
        head_ip = str(self.daq_conf.head_node_ip_addr)
        reachable = nx.descendants(graph, head_ip) | {head_ip}
        all_nodes = set(graph.nodes)
        orphans = all_nodes - reachable
        
        critical_orphans = [n for n in orphans if graph.nodes[n].get("role") in ["quabo", "module"]]
        if critical_orphans:
            self.report.add_test("Topology Reachability", "ERROR", 
                                 f"Hardware nodes unreachable from Head Node: {critical_orphans}")
        else:
            self.report.add_test("Topology Reachability", "PASS", "All hardware reachable from Head Node.")

        # 3. Check for DAQ Node Bottlenecks (more than k modules)
        module_limit = self.daq_conf.daq_node_module_limit or MAX_MODULES_PER_DAQ_NODE
        daq_nodes = [n for n, d in graph.nodes(data=True) if d.get("role") == "daqnode"]
        for node_ip in daq_nodes:
            # find successors with role "module"
            successors = nx.descendants(graph, node_ip)
            modules = [n for n in successors if graph.nodes[n].get("role") == "module"]
            if len(modules) > module_limit:
                self.report.add_test("DAQ Node Bottleneck", "WARN", 
                                     f"DAQ Node {node_ip} handles {len(modules)} modules (> {module_limit} limit). Processing may be delayed.")
            else:
                self.report.add_test("DAQ Node Bottleneck", "PASS", f"DAQ Node {node_ip} load balanced.")

        # 4. Check for Control Loops (Must be a DAG)
        graph_no_self_loops = graph.copy()
        graph_no_self_loops.remove_edges_from(nx.selfloop_edges(graph_no_self_loops))
        if not nx.is_directed_acyclic_graph(graph_no_self_loops):
            cycles = list(nx.simple_cycles(graph_no_self_loops))
            self.report.add_test("Control Loop Check", "ERROR", f"Infinite control loops detected: {cycles}")
        else:
            self.report.add_test("Control Loop Check", "PASS", "No control loops detected.")

    def _check_daq_module_subnet_coherence(self) -> None:
        """
        Verify that DAQ nodes and the modules they service are in the same subnet.
        At high data rates, going through a router (gateway) between DAQ and Quabo is invalid.
        """
        if not self.daq_conf or not self.net_conf or not self.obs_conf:
            return

        # Map IPs to their Gateway IPs from the network config
        daq_gw_map = {str(n.ip_addr): (str(n.port_forwarding.gw_ip) if n.port_forwarding and n.port_forwarding.status else None) 
                      for n in self.net_conf.daq_nodes}
        mod_gw_map = {str(m.ip_addr): (str(m.port_forwarding.gw_ip) if m.port_forwarding and m.port_forwarding.status else None) 
                      for m in self.net_conf.modules}

        errors = []
        for daq in self.daq_conf.daq_nodes:
            daq_ip = str(daq.ip_addr)
            daq_gw = daq_gw_map.get(daq_ip)
            
            for mid in daq.module_ids:
                # Find the module's physical IP in obs_config
                module = next((m for d in self.obs_conf.domes for m in d.modules if m.id == mid), None)
                if not module:
                    continue
                
                mod_ip = str(module.ip_addr)
                mod_gw = mod_gw_map.get(mod_ip)
                
                if daq_gw != mod_gw:
                    errors.append(f"DAQ {daq_ip} (GW: {daq_gw}) cannot handle Module {mid} at {mod_ip} (GW: {mod_gw}). Subnet mismatch.")

        if errors:
            self.report.add_test("Subnet Coherence", "ERROR", f"DAQ/Quabo subnet mismatches detected: {errors}")
        else:
            self.report.add_test("Subnet Coherence", "PASS", "All DAQ nodes and Quabos reside in coherent subnets.")
