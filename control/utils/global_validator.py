"""
global_validator.py

Executes Tier-2 cross-configuration validations to ensure physical, network,
and hardware states are cohesive across the entire PanoSETI observatory.
"""

import math
from typing import Dict, Any, List
from haversine import haversine, Unit
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MAX_DOME_BASELINE_KM = 1

class ValidationReport:
    """Aggregates errors and warnings for a unified pre-flight report."""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def print_report(self):
        """Prints a nicely formatted table of all findings to the console."""
        if not self.errors and not self.warnings:
            console.print(Panel("[bold green]All Global Pre-Flight Checks Passed![/bold green]", border_style="green"))
            return

        table = Table(title="Global Configuration Validation Report", show_lines=True)
        table.add_column("Severity", justify="center", style="bold")
        table.add_column("Message")

        for err in self.errors:
            table.add_row("[red]ERROR[/red]", f"[red]{err}[/red]")
        for warn in self.warnings:
            table.add_row("[yellow]WARNING[/yellow]", f"[yellow]{warn}[/yellow]")

        console.print(table)


class GlobalConfigValidator:
    """
    Cross-references all loaded PanoSETI configurations.
    Methods starting with '_check_' are automatically executed.
    """
    def __init__(self, validated_configs: Dict[str, Any]):
        # These are assumed to be the raw dictionaries that have already
        # passed Tier-1 Pydantic validation.
        self.obs_conf = validated_configs.get('obs', {})
        self.data_conf = validated_configs.get('data', {})
        self.daq_conf = validated_configs.get('daq', {})
        self.net_conf = validated_configs.get('network', {})
        self.firmware_conf = validated_configs.get('firmware', {})
        self.report = ValidationReport()

    def validate_all_rules(self) -> bool:
        """Executes all rule methods and prints the report."""
        rule_methods = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and func.startswith("_check_")]
        for rule in rule_methods:
            rule()

        self.report.print_report()
        return not self.report.has_errors()

    def _check_science_guardrails(self):
        """Warns if artificial signals are enabled during a non-engineering run."""
        if not self.data_conf:
            return

        run_type = self.data_conf.get('run_type', '').lower()
        if "eng" not in run_type:
            flash_on = 'flash_params' in self.data_conf
            stim_on = 'stim_params' in self.data_conf

            if flash_on or stim_on:
                self.report.add_warning(
                    f"Non-engineering run_type '{run_type}' detected, but flash_params or stim_params are ENABLED. "
                    "This will inject artificial signals into science data."
                )

    def _check_geospatial_coherence(self):
        """Ensures all domes baselines in the observatory are at most {MAX_DOME_BASELINE_KM} kilometers."""
        if not self.obs_conf or 'domes' not in self.obs_conf:
            return

        domes = self.obs_conf['domes']
        if len(domes) < 2:
            return

        coords = [(d['name'], d['obslat'], d['obslon']) for d in domes if 'obslat' in d and 'obslon' in d]

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                name1, lat1, lon1 = coords[i]
                name2, lat2, lon2 = coords[j]

                # Calculate distance in kilometers
                distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)

                if distance > MAX_DOME_BASELINE_KM:
                    self.report.add_error(
                        f"Geospatial anomaly: Domes '{name1}'\t and '{name2}'\t are \t{distance:.3f} km apart "
                        f"(> {MAX_DOME_BASELINE_KM:.3f}km limit). "
                        "Check for missing errors in the GPS coordinates."
                    )

    def _check_network_tunneling(self):
        """Ensures every module in obs_config has explicit mapping if port forwarding is needed."""
        if not self.obs_conf or not self.net_conf:
            return

        obs_ips = []
        for dome in self.obs_conf.get('domes', []):
            for mod in dome.get('modules', []):
                obs_ips.append(mod.get('ip_addr'))

        net_modules = self.net_conf.get('modules', [])
        net_mapped_ips = {m.get('ip_addr') for m in net_modules if m.get('port_forwarding', {}).get('status') is True}

        # If network tunneling is entirely empty, assume direct local network and skip
        if not net_modules:
            return

        for ip in obs_ips:
            if ip not in net_mapped_ips:
                self.report.add_warning(
                    f"Module IP {ip} defined in obs_config but lacks active port_forwarding in network_config. "
                    "If on a remote subnet, commands to this module will fail silently."
                )

    def _check_hardware_firmware(self):
        """Ensures firmware binaries exist for the hardware versions specified in the domes."""
        if not self.obs_conf or not self.firmware_conf:
            return

        required_hw = set()
        for dome in self.obs_conf.get('domes', []):
            for mod in dome.get('modules', []):
                qv = mod.get('quabo_version')
                if isinstance(qv, list):
                    required_hw.update(qv)
                elif isinstance(qv, str):
                    required_hw.add(qv)

        for hw_version in required_hw:
            if hw_version not in self.firmware_conf:
                self.report.add_error(
                    f"Hardware mismatch: obs_config requires '{hw_version}' firmware, but it is missing from firmware.json."
                )

    def _check_overvoltage_consensus(self):
        """Ensures obs_config and data_config agree on the physical overvoltage."""
        if not self.obs_conf or not self.data_conf:
            return

        obs_ov = self.obs_conf.get('detector_overvoltage')
        data_ov = self.data_conf.get('detector_overvoltage')

        if obs_ov is not None and data_ov is not None and obs_ov != data_ov:
            self.report.add_error(
                f"Overvoltage mismatch: obs_config says {obs_ov}V, but data_config says {data_ov}V. "
                "This will result in invalid calibrations."
            )