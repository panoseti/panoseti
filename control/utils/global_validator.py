"""
global_validator.py

Executes Tier-2 cross-configuration validations to ensure physical, network,
and hardware states are cohesive across the entire PanoSETI observatory.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from haversine import Unit, haversine
from rich.console import Console
from rich.table import Table

from .pydantic_config_models import (
    DaqConfigValidator,
    DataConfigValidator,
    FirmwareConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
)

console = Console()

MAX_DOME_BASELINE_KM = 2

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


class GlobalConfigValidator:
    """Executes cross-configuration validation rules for the observatory.
    
    This validator ensures that individual configuration files (obs, data, 
    daq, network, firmware) are mutually consistent and physically plausible.
    """

    def __init__(self, validated_configs: dict[str, Any]):
        """Initialize the validator with validated configuration models.

        Args:
            validated_configs: Dictionary containing Pydantic model instances 
                               for each configuration type.
        """
        self.obs_conf: ObsConfigValidator = validated_configs.get('obs') # type: ignore
        self.data_conf: DataConfigValidator = validated_configs.get('data') # type: ignore
        self.daq_conf: DaqConfigValidator = validated_configs.get('daq') # type: ignore
        self.net_conf: NetworkConfigValidator = validated_configs.get('network') # type: ignore
        self.firmware_conf: FirmwareConfigValidator = validated_configs.get('firmware') # type: ignore
        self.report = ValidationReport()

    def validate_all_rules(self) -> bool:
        """Execute all validation methods prefixed with '_check_'.

        Returns:
            True if all rules passed (including warnings), False if any ERROR occurred.
        """
        rule_methods = [getattr(self, func) for func in dir(self) if
                        callable(getattr(self, func)) and func.startswith("_check_")]
        for rule in rule_methods:
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
        self.report.add_test("Science Guardrails", "PASS", f"Run type: {run_type}")

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
        """Verify that firmware binaries exist for all active hardware types (BGA/QFP)."""
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

        firmware_keys = set((self.firmware_conf.model_extra or {}).keys())
        missing = required_hw - firmware_keys
        if missing:
            self.report.add_test("Firmware Verification", "ERROR", f"Missing binaries for HW types: {missing}")
        else:
            self.report.add_test("Firmware Verification", "PASS", "Binaries exist for all active hardware.")

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

    # --- NEW TEST 2: DAQ Assignment Overlap Check ---
    def _check_daq_assignment_overlap(self) -> None:
        """Ensure a single module ID is not handled by multiple DAQ nodes."""
        if not self.daq_conf:
            return
        from .config_file import expand_ranges
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
        if not head_dir or not os.path.exists(head_dir):
            self.report.add_test("Headnode Disk Space", "ERROR", f"Path '{head_dir}' missing or unreachable.")
            return

        _total, _used, free = shutil.disk_usage(head_dir)
        free_tb = free / (1024 ** 4)

        tb_per_hr, est_total, formula = self._estimate_data_usage()

        msg = f"Available: {free_tb:.2f} TB. Est: {tb_per_hr:.3f} TB/hr. Formula: {formula}"
        if est_total > 0 and (free_tb - est_total) <= 0:
            self.report.add_test("Headnode Disk Space", "ERROR", f"INSUFFICIENT SPACE! {msg}")
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
