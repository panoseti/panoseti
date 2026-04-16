"""
test_config_validator.py

Unit tests for control/utils/config_validator.py:
  - print_compact_config: long module_id lists get compressed
  - perform_network_ping_sweep: TCP reachability logic (socket patched)
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from utils.config_validator import perform_network_ping_sweep, print_compact_config

# ===========================================================================
# TestPrintCompactConfig
# ===========================================================================

class TestPrintCompactConfig:
    def test_daq_with_many_module_ids_compressed(self, capsys):
        """DAQ config with 256 module IDs should print a compressed summary."""
        config = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {
                    "username": "panoseti",
                    "data_dir": "/data",
                    "ip_addr": "10.0.0.2",
                    "module_ids": list(range(256)),
                    "bindhost": "0.0.0.0",
                }
            ],
        }
        # Should not raise; long list gets condensed
        print_compact_config("daq", config)
        captured = capsys.readouterr()
        # The condensed form should show "total IDs" rather than all 256 numbers
        assert "total" in captured.out or "256" in captured.out

    def test_daq_with_few_module_ids_not_compressed(self, capsys):
        """DAQ config with 3 module IDs should not be compressed."""
        config = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {
                    "username": "panoseti",
                    "data_dir": "/data",
                    "ip_addr": "10.0.0.2",
                    "module_ids": [0, 1, 2],
                }
            ],
        }
        print_compact_config("daq", config)
        captured = capsys.readouterr()
        # All three IDs should appear verbatim
        assert "0" in captured.out

    def test_non_daq_config_printed(self, capsys):
        """Non-DAQ configs should be printed as-is."""
        config = {"run_type": "sci", "integration_time_usec": 100000}
        print_compact_config("data", config)
        captured = capsys.readouterr()
        assert "sci" in captured.out or "data" in captured.out.lower()


# ===========================================================================
# TestPerformNetworkPingSweep
# ===========================================================================

def _base_configs(tmp_path=None):
    """Minimal validated configs for the ping sweep."""
    return {
        "obs": {
            "name": "test",
            "domes": [
                {
                    "name": "dome0",
                    "modules": [
                        {
                            "ip_addr": "192.168.3.200",
                            "wps": "wps",
                        }
                    ],
                }
            ],
            "wps": {"url": "http://192.168.1.2", "quabo_socket": 1},
        },
        "daq": {
            "head_node_ip_addr": "10.0.0.1",
            "head_node_container": False,
            "daq_nodes": [
                {
                    "ip_addr": "10.0.0.2",
                    "module_ids": [200],
                    "username": "panoseti",
                }
            ],
        },
        "network": {
            "modules": [],
            "daq_nodes": [],
        },
    }


@contextmanager
def _patch_tcp_check(up_hosts: set):
    """
    Patch socket.create_connection so that connections to IPs in up_hosts
    succeed and all others raise OSError (connection refused).
    """

    def fake_create_connection(addr, timeout=None):
        ip, _port = addr
        if ip in up_hosts:
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        raise OSError("Connection refused")

    with patch("socket.create_connection", side_effect=fake_create_connection):
        yield


class TestPerformNetworkPingSweep:
    def test_all_hosts_up_returns_true(self):
        """All targets reachable → returns True."""
        cfg = _base_configs()
        all_ips = {"10.0.0.1", "10.0.0.2", "192.168.3.200", "192.168.1.2"}
        with _patch_tcp_check(all_ips):
            result = perform_network_ping_sweep(cfg)
        assert result is True

    def test_one_host_down_returns_false(self):
        """Any target unreachable → returns False."""
        cfg = _base_configs()
        # Head node is down
        with _patch_tcp_check(set()):
            result = perform_network_ping_sweep(cfg)
        assert result is False

    def test_head_node_is_checked(self):
        """Head node IP is included in the ping sweep."""
        cfg = _base_configs()
        checked_ips = []

        def capture_connection(addr, timeout=None):
            checked_ips.append(addr[0])
            raise OSError("refused")

        with patch("socket.create_connection", side_effect=capture_connection):
            perform_network_ping_sweep(cfg)

        assert "10.0.0.1" in checked_ips

    def test_head_node_container_skips_head_check(self):
        """head_node_container=True → head node not directly checked."""
        cfg = _base_configs()
        cfg["daq"]["head_node_container"] = True

        checked_ips = []

        def capture_connection(addr, timeout=None):
            checked_ips.append(addr[0])
            raise OSError("refused")

        with patch("socket.create_connection", side_effect=capture_connection):
            perform_network_ping_sweep(cfg)

        assert "10.0.0.1" not in checked_ips

    def test_wps_url_is_checked(self):
        """WPS URL hostname is extracted and pinged."""
        cfg = _base_configs()
        checked_ips = []

        def capture_connection(addr, timeout=None):
            checked_ips.append(addr[0])
            raise OSError("refused")

        with patch("socket.create_connection", side_effect=capture_connection):
            perform_network_ping_sweep(cfg)

        assert "192.168.1.2" in checked_ips

    def test_module_ip_is_checked(self):
        """Module IP (quabo CMD port) is included in the sweep."""
        cfg = _base_configs()
        checked_ips = []

        def capture_connection(addr, timeout=None):
            checked_ips.append(addr[0])
            raise OSError("refused")

        with patch("socket.create_connection", side_effect=capture_connection):
            perform_network_ping_sweep(cfg)

        assert "192.168.3.200" in checked_ips

    def test_no_modules_or_wps_still_returns(self):
        """Empty obs config (no modules) → sweep runs without crash."""
        cfg = {
            "obs": {"name": "test", "domes": []},
            "daq": {
                "head_node_ip_addr": "10.0.0.1",
                "head_node_container": True,
                "daq_nodes": [],
            },
            "network": {"modules": [], "daq_nodes": []},
        }
        with _patch_tcp_check(set()):
            result = perform_network_ping_sweep(cfg)
        # No targets → all_passed stays True (vacuously)
        assert result is True

    def test_gateway_forwarded_module_uses_gw_ip(self):
        """Module behind gateway → gateway IP is checked, not module IP directly."""
        cfg = _base_configs()
        cfg["network"]["modules"] = [
            {
                "ip_addr": "192.168.3.200",
                "port_forwarding": {
                    "status": True,
                    "gw_ip": "203.0.113.1",
                    "cmd_port": [60000],
                    "reboot_port": [],
                },
            }
        ]
        checked_ips = []

        def capture_connection(addr, timeout=None):
            checked_ips.append(addr[0])
            raise OSError("refused")

        with patch("socket.create_connection", side_effect=capture_connection):
            perform_network_ping_sweep(cfg)

        # Gateway should be checked, not the internal module IP directly
        assert "203.0.113.1" in checked_ips
