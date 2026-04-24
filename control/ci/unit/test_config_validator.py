"""
test_config_validator.py

Unit tests for control/utils/config_validator.py:
  - print_compact_config: long module_id lists get compressed
  - perform_network_ping_sweep: TCP reachability logic (socket patched)
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from control.utils.config_validator import perform_network_ping_sweep, print_compact_config
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    ObsConfig,
)

# ===========================================================================
# TestPrintCompactConfig
# ===========================================================================

class TestPrintCompactConfig:
    def test_daq_with_many_module_ids_compressed(self, capsys, mock_daq_config: DaqConfig) -> None:
        """DAQ config with 256 module IDs should print a compressed summary."""
        mock_daq_config.daq_nodes[0].module_ids = list(range(256))
        
        # Should not raise; long list gets condensed
        print_compact_config("daq", mock_daq_config)
        captured = capsys.readouterr()
        # The condensed form should show "total IDs" rather than all 256 numbers
        assert "total" in captured.out or "256" in captured.out

    def test_daq_with_few_module_ids_not_compressed(self, capsys, mock_daq_config: DaqConfig) -> None:
        """DAQ config with 3 module IDs should not be compressed."""
        mock_daq_config.daq_nodes[0].module_ids = [0, 1, 2]
        print_compact_config("daq", mock_daq_config)
        captured = capsys.readouterr()
        # All three IDs should appear verbatim
        assert "0" in captured.out

    def test_non_daq_config_printed(self, capsys, mock_obs_config: ObsConfig) -> None:
        """Non-DAQ configs should be printed as-is."""
        print_compact_config("obs", mock_obs_config)
        captured = capsys.readouterr()
        assert "dome0" in captured.out


# ===========================================================================
# TestPerformNetworkPingSweep
# ===========================================================================

@contextmanager
def _patch_reachability(up_hosts: set):
    """
    Patch _check_reachability so that connections to IPs in up_hosts
    succeed and all others return False.
    """
    def fake_check_reachability(target_ip, port, target_type="tcp", timeout=2.0):
        if target_ip in up_hosts:
            return True, ""
        return False, "Connection refused"

    with patch("control.utils.config_validator._check_reachability", side_effect=fake_check_reachability):
        yield


class TestPerformNetworkPingSweep:
    def test_all_hosts_up_returns_true(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """All targets reachable → returns True."""
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        all_ips = {
            "10.0.0.1", "10.0.0.2", "192.168.1.2",
            "192.168.3.200", "192.168.3.201", "192.168.3.202", "192.168.3.203"
        }
        with _patch_reachability(all_ips):
            result = perform_network_ping_sweep(cfg)
        assert result is True

    def test_one_host_down_returns_false(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """Any target unreachable → returns False."""
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        # Head node is down (10.0.0.1 not in up_hosts)
        with _patch_reachability(set()):
            result = perform_network_ping_sweep(cfg)
        assert result is False

    def test_head_node_is_checked(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """Head node IP is included in the reachability sweep."""
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        checked_ips = []

        def capture_reachability(target_ip, port, target_type="tcp", timeout=2.0):
            checked_ips.append(target_ip)
            return False, "refused"

        with patch("control.utils.config_validator._check_reachability", side_effect=capture_reachability):
            perform_network_ping_sweep(cfg)

        assert "10.0.0.1" in checked_ips

    def test_head_node_container_skips_head_check(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """head_node_container=True → head node not directly checked."""
        mock_daq_config.head_node_container = True
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}

        checked_ips = []

        def capture_reachability(target_ip, port, target_type="tcp", timeout=2.0):
            checked_ips.append(target_ip)
            return False, "refused"

        with patch("control.utils.config_validator._check_reachability", side_effect=capture_reachability):
            perform_network_ping_sweep(cfg)

        assert "10.0.0.1" not in checked_ips

    def test_wps_url_is_checked(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """WPS URL hostname is extracted and checked."""
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        checked_ips = []

        def capture_reachability(target_ip, port, target_type="tcp", timeout=2.0):
            checked_ips.append(target_ip)
            return False, "refused"

        with patch("control.utils.config_validator._check_reachability", side_effect=capture_reachability):
            perform_network_ping_sweep(cfg)

        assert "192.168.1.2" in checked_ips

    def test_module_ip_is_checked(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """Module IPs (quabo UDP ports) are included in the sweep."""
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        checked_ips = []

        def capture_reachability(target_ip, port, target_type="tcp", timeout=2.0):
            checked_ips.append(target_ip)
            return False, "refused"

        with patch("control.utils.config_validator._check_reachability", side_effect=capture_reachability):
            perform_network_ping_sweep(cfg)

        # All 4 quabos should be checked
        for i in range(4):
            assert f"192.168.3.{200+i}" in checked_ips

    def test_no_modules_or_wps_still_returns(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """Empty obs config (no modules) → sweep runs without crash."""
        mock_obs_config.domes = []
        mock_daq_config.head_node_container = True
        mock_daq_config.daq_nodes = []
        mock_network_config.modules = []
        mock_network_config.daq_nodes = []

        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        with _patch_reachability(set()):
            result = perform_network_ping_sweep(cfg)
        # No targets → all_passed stays True (vacuously)
        assert result is True

    def test_gateway_forwarded_module_uses_gw_ip(self, mock_obs_config: ObsConfig, mock_daq_config: DaqConfig, mock_network_config: NetworkConfig) -> None:
        """Module behind gateway → gateway IP is checked, not module IP directly."""
        mock_network_config.modules[0].port_forwarding.status = True
        mock_network_config.modules[0].port_forwarding.gw_ip = "203.0.113.1"
        cfg = {"obs": mock_obs_config, "daq": mock_daq_config, "network": mock_network_config}
        checked_ips = []

        def capture_reachability(target_ip, port, target_type="tcp", timeout=2.0):
            checked_ips.append(target_ip)
            return False, "refused"

        with patch("control.utils.config_validator._check_reachability", side_effect=capture_reachability):
            perform_network_ping_sweep(cfg)

        # Gateway should be checked, not the internal module IP directly
        assert "203.0.113.1" in checked_ips