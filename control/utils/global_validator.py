"""
global_validator.py

Executes Tier-2 cross-configuration validations to ensure physical, network,
and hardware states are cohesive across the entire PanoSETI observatory.
"""

import os
import shutil
import subprocess
from typing import Dict, Any, List, Tuple
from haversine import haversine, Unit
from rich.console import Console
from rich.table import Table

console = Console()

MAX_DOME_BASELINE_KM = 1

class ValidationReport:
    """Aggregates tests for a unified pre-flight report."""

    def __init__(self):
        self.tests: List[Dict[str, str]] = []
        self.has_errors = False

    def add_test(self, name: str, status: str, info: str = ""):
        self.tests.append({"name": name, "status": status, "info": info})
        if status == "ERROR":
            self.has_errors = True

    def print_report(self):
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
    def __init__(self, validated_configs: Dict[str, Any]):
        self.obs_conf = validated_configs.get('obs', {})
        self.data_conf = validated_configs.get('data', {})
        self.daq_conf = validated_configs.get('daq', {})
        self.net_conf = validated_configs.get('network', {})
        self.firmware_conf = validated_configs.get('firmware', {})
        self.report = ValidationReport()

    def validate_all_rules(self) -> bool:
        rule_methods = [getattr(self, func) for func in dir(self) if
                        callable(getattr(self, func)) and func.startswith("_check_")]
        for rule in rule_methods:
            rule()
        self.report.print_report()
        return not self.report.has_errors

    def _check_science_guardrails(self):
        run_type = self.data_conf.get('run_type', '').lower()
        if "eng" not in run_type:
            flash_on = 'flash_params' in self.data_conf
            stim_on = 'stim_params' in self.data_conf
            if flash_on or stim_on:
                self.report.add_test("Science Guardrails", "WARN",
                                     f"Run type '{run_type}' has flash/stim enabled. Artificial signals will be injected.")
                return
        self.report.add_test("Science Guardrails", "PASS", f"Run type: {run_type}")

    def _check_geospatial_coherence(self):
        domes = self.obs_conf.get('domes', [])
        if len(domes) < 2:
            self.report.add_test("Geospatial Coherence", "PASS", "Only one dome defined.")
            return

        coords = [(d['name'], d.get('obslat'), d.get('obslon')) for d in domes if 'obslat' in d]
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

    def _check_network_tunneling(self):
        obs_ips = {m.get('ip_addr') for d in self.obs_conf.get('domes', []) for m in d.get('modules', [])}
        net_mapped_ips = {m.get('ip_addr') for m in self.net_conf.get('modules', []) if
                          m.get('port_forwarding', {}).get('status')}

        missing = obs_ips - net_mapped_ips
        if missing and self.net_conf.get('modules'):
            self.report.add_test("Network Tunneling Mapping", "WARN",
                                 f"Modules lacking port forwarding: {', '.join(missing)}")
        else:
            self.report.add_test("Network Tunneling Mapping", "PASS", "All modules accounted for in routing.")

    def _check_hardware_firmware(self):
        required_hw = set()
        for d in self.obs_conf.get('domes', []):
            for m in d.get('modules', []):
                qv = m.get('quabo_version')
                if isinstance(qv, list):
                    required_hw.update(qv)
                else:
                    required_hw.add(qv)

        missing = required_hw - set(self.firmware_conf.keys())
        if missing:
            self.report.add_test("Firmware Verification", "ERROR", f"Missing binaries for HW types: {missing}")
        else:
            self.report.add_test("Firmware Verification", "PASS", "Binaries exist for all active hardware.")

    def _check_overvoltage_consensus(self):
        obs_ov = self.obs_conf.get('detector_overvoltage')
        data_ov = self.data_conf.get('detector_overvoltage')
        if obs_ov is not None and data_ov is not None and obs_ov != data_ov:
            self.report.add_test("Overvoltage Consensus", "ERROR",
                                 f"obs_config ({obs_ov}V) != data_config ({data_ov}V)")
        else:
            self.report.add_test("Overvoltage Consensus", "PASS", f"Voltage aligned at {obs_ov}V")

    def _check_port_collisions(self):
        """Ensures multiple modules sharing a Gateway IP do not use overlapping forwarded ports."""
        gw_ports = {}
        for m in self.net_conf.get('modules', []):
            pf = m.get('port_forwarding', {})
            if pf.get('status'):
                gw = pf.get('gw_ip')
                ports = pf.get('cmd_port', []) + pf.get('reboot_port', [])
                if gw not in gw_ports:
                    gw_ports[gw] = set()
                for p in ports:
                    if p in gw_ports[gw]:
                        self.report.add_test("Port Collision", "ERROR",
                                             f"Gateway {gw} has multiple modules attempting to forward port {p}.")
                        return
                    gw_ports[gw].add(p)
        self.report.add_test("Port Collision", "PASS", "No forwarded port overlaps detected on gateways.")

    # --- NEW TEST 2: DAQ Assignment Overlap Check ---
    def _check_daq_assignment_overlap(self):
        """Ensures a single module ID is not being actively listened to by multiple DAQ nodes."""
        from .config_file import _expand_module_ids
        seen_ids = set()
        for daq in self.daq_conf.get('daq_nodes', []):
            ids = _expand_module_ids(daq.get('module_ids', ''))
            overlap = seen_ids.intersection(ids)
            if overlap:
                self.report.add_test("DAQ Overlap", "ERROR",
                                     f"Module IDs {overlap} are assigned to multiple DAQ nodes!")
                return
            seen_ids.update(ids)
        self.report.add_test("DAQ Overlap", "PASS", "No overlapping module_ids across DAQ nodes.")

    def _estimate_data_usage(self) -> Tuple[float, float, str]:
        """Returns (TB_per_hour, total_estimated_TB, formula_string)"""
        num_modules = sum(len(d.get('modules', [])) for d in self.obs_conf.get('domes', []))
        img_conf = self.data_conf.get('image', {})
        if num_modules == 0 or not img_conf: return 0.0, 0.0, "N/A"

        int_usec = img_conf.get('integration_time_usec', 100000)
        nsum = img_conf.get('nsum', 1)
        bytes_pp = img_conf.get('quabo_sample_size', 16) / 8

        fps = 1_000_000 / (int_usec * nsum)
        tb_per_hr = (fps * 4 * 1024 * bytes_pp * 3600 * num_modules) / (1024 ** 4)
        total_tb = tb_per_hr * 8  # Assume 8 hour run

        formula = f"({fps:.1f}frame/sec * 4quabo * 1024px/quabo * {bytes_pp}B * 3600sec * {num_modules}mod) / 1TiB"
        return tb_per_hr, total_tb, formula

    def _check_headnode_disk_space(self):
        head_dir = self.daq_conf.get('head_node_data_dir')
        if not head_dir or not os.path.exists(head_dir):
            self.report.add_test("Headnode Disk Space", "WARN", f"Path '{head_dir}' missing or unreachable.")
            return

        total, used, free = shutil.disk_usage(head_dir)
        free_tb = free / (1024 ** 4)

        tb_per_hr, est_total, formula = self._estimate_data_usage()

        msg = f"Available: {free_tb:.2f} TB. Est: {tb_per_hr:.3f} TB/hr. Formula: {formula}"
        if est_total > 0 and (free_tb - est_total) <= 0:
            self.report.add_test("Headnode Disk Space", "ERROR", f"INSUFFICIENT SPACE! {msg}")
        elif free_tb < 1.0:
            self.report.add_test("Headnode Disk Space", "WARN", f"Low Space! {msg}")
        else:
            self.report.add_test("Headnode Disk Space", "PASS", msg)

    def _check_daqnode_disk_space(self):
        """Uses SSH to check the specific data volume on each DAQ node."""
        results = []
        has_error = False
        tb_per_hr, est_total, _ = self._estimate_data_usage()

        for daq in self.daq_conf.get('daq_nodes', []):
            ip = daq.get('ip_addr')
            usr = daq.get('username', 'panoseti')
            data_dir = daq.get('data_dir')

            # Determine SSH routing (direct vs gateway)
            ssh_ip = ip
            ssh_port = 22
            for net_daq in self.net_conf.get('daq_nodes', []):
                if net_daq.get('ip_addr') == ip and net_daq.get('port_forwarding', {}).get('status'):
                    ssh_ip = net_daq['port_forwarding']['gw_ip']
                    ssh_port = net_daq['port_forwarding'].get('port', 22)
                    break

            cmd = ['ssh', '-p', str(ssh_port), '-o', 'ConnectTimeout=2', '-o', 'BatchMode=yes', f'{usr}@{ssh_ip}', 'df',
                   '-k', data_dir]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    lines = res.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split()
                        free_kb = int(parts[3])
                        free_tb = free_kb / (1024 ** 3)

                        if est_total > 0 and (free_tb - est_total) <= 0:
                            results.append(f"{ip}: {free_tb:.2f} TB (INSUFFICIENT)")
                            has_error = True
                        elif free_tb < 1.0:
                            results.append(f"{ip}: {free_tb:.2f} TB (LOW)")
                        else:
                            results.append(f"{ip}: {free_tb:.2f} TB (OK)")
                    else:
                        results.append(f"{ip}: Path Not Found")
                        has_error = True
                else:
                    results.append(f"{ip}: SSH Failed")
                    has_error = True
            except Exception:
                results.append(f"{ip}: SSH Error")

        if not results:
            self.report.add_test("DAQ Node Disk Space", "PASS", "No DAQ nodes configured.")
            return

        detail_str = " | ".join(results)
        if has_error:
            self.report.add_test("DAQ Node Disk Space", "ERROR", detail_str)
        else:
            self.report.add_test("DAQ Node Disk Space", "PASS", detail_str)

    def _check_wps_references(self):
        """Ensures that all Web Power Switches referenced by modules are defined in obs_config."""
        if not self.obs_conf: return

        missing_wps = set()
        for d in self.obs_conf.get('domes', []):
            for m in d.get('modules', []):
                # The default is 'wps' if not specified
                wps_name = m.get('wps', 'wps')
                if wps_name not in self.obs_conf:
                    missing_wps.add(wps_name)

        if missing_wps:
            self.report.add_test("WPS Reference Map", "ERROR",
                                 f"Modules reference undefined WPS units: {', '.join(missing_wps)}")
        else:
            self.report.add_test("WPS Reference Map", "PASS", "All referenced WPS units exist in obs_config.")

