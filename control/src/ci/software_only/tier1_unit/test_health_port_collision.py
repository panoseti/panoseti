"""
test_health_port_collision.py

Unit tests for control.health._check_port_collision() -- pure, no-network
detection of a co-located head+DAQ node gRPC port collision.

Precedence under test: network_config.json's headnode.grpc_port / a node's
own grpc_port (via attach_daq_config) > the HEADNODE_GRPC_PORT/DAQNODE_GRPC_PORT
env vars > the 50051 default -- same resolution _check_grpc_headnode() and
_check_grpc_daq_nodes() use, so this check can't disagree with what those
two actually probe.
"""
from __future__ import annotations

import pytest

import control.utils.config_file as config_file_mod
from control.health import _check_port_collision
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    NetworkConfig,
    NetworkDaqNode,
    NetworkHeadnode,
    PortForwarding,
)

_PORT_ENV_VARS = ("HEADNODE_GRPC_PORT", "DAQNODE_GRPC_PORT", "DAQ_DATA_GATEWAY_PORT", "GRPC_PORT")


@pytest.fixture(autouse=True)
def _clean_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PORT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _co_located_daq_config(**node_overrides: object) -> DaqConfig:
    """A single DAQ node co-located with the head node (head_node_container=True
    is the deterministic, network-free way is_local() reports True in CI --
    see control/CLAUDE.md's "Validation Leniency" note).
    """
    node_defaults: dict = dict(
        username="panoseti",
        data_dir="/data",
        ip_addr="10.0.0.1",
        module_ids=[0],
    )
    node_defaults.update(node_overrides)
    return DaqConfig(
        head_node_data_dir="/data/head",
        head_node_ip_addr="10.0.0.1",
        head_node_container=True,
        daq_nodes=[DaqNode(**node_defaults)],
    )


def _patch_configs(monkeypatch: pytest.MonkeyPatch, daq_config: DaqConfig, network_config: NetworkConfig) -> None:
    monkeypatch.setattr(config_file_mod, "get_daq_config", lambda *a, **kw: daq_config)
    monkeypatch.setattr(config_file_mod, "get_network_config", lambda *a, **kw: network_config)


class TestPortCollisionPrecedence:
    def test_env_vars_only_no_collision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50051")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50052")
        _patch_configs(monkeypatch, _co_located_daq_config(), NetworkConfig())

        results = _check_port_collision()

        assert len(results) == 1
        _name, ok, detail = results[0]
        assert ok is True
        assert "50051" in detail and "50052" in detail

    def test_env_vars_only_collision_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50051")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50051")
        _patch_configs(monkeypatch, _co_located_daq_config(), NetworkConfig())

        results = _check_port_collision()

        assert len(results) == 1
        _name, ok, detail = results[0]
        assert ok is False
        assert "50051" in detail

    def test_network_config_headnode_port_takes_precedence_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """headnode.grpc_port=60051 in network_config.json must win over
        HEADNODE_GRPC_PORT=50051 -- resolving a *false* collision (env vars
        alone would show 50051 == 50051) once the explicit override applies.
        """
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50051")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50051")
        network_config = NetworkConfig(headnode=NetworkHeadnode(grpc_port=60051))
        _patch_configs(monkeypatch, _co_located_daq_config(), network_config)

        results = _check_port_collision()

        assert len(results) == 1
        _name, ok, detail = results[0]
        assert ok is True
        assert "60051" in detail and "50051" in detail

    def test_network_config_daq_node_port_takes_precedence_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A node's own network_config.json grpc_port must win over DAQNODE_GRPC_PORT."""
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50051")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50051")
        network_config = NetworkConfig(
            daq_nodes=[
                NetworkDaqNode(
                    ip_addr="10.0.0.1",  # type: ignore
                    grpc_port=60052,
                    port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
                )
            ]
        )
        _patch_configs(monkeypatch, _co_located_daq_config(), network_config)

        results = _check_port_collision()

        assert len(results) == 1
        _name, ok, detail = results[0]
        assert ok is True
        assert "60052" in detail

    def test_network_config_still_collides_when_both_set_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """network_config.json's precedence doesn't hide a real collision it introduces itself."""
        network_config = NetworkConfig(
            headnode=NetworkHeadnode(grpc_port=60051),
            daq_nodes=[
                NetworkDaqNode(
                    ip_addr="10.0.0.1",  # type: ignore
                    grpc_port=60051,
                    port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
                )
            ],
        )
        _patch_configs(monkeypatch, _co_located_daq_config(), network_config)

        results = _check_port_collision()

        assert len(results) == 1
        _name, ok, _detail = results[0]
        assert ok is False

    def test_daq_config_own_explicit_grpc_port_still_wins_over_network_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """daq_config.json's own per-node override (highest precedence in
        attach_daq_config()) still beats network_config.json's value here too.
        """
        network_config = NetworkConfig(
            daq_nodes=[
                NetworkDaqNode(
                    ip_addr="10.0.0.1",  # type: ignore
                    grpc_port=60052,
                    port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
                )
            ]
        )
        daq_config = _co_located_daq_config(grpc_port=60099)
        _patch_configs(monkeypatch, daq_config, network_config)

        results = _check_port_collision()

        assert len(results) == 1
        _name, _ok, detail = results[0]
        assert "60099" in detail

    def test_non_co_located_node_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A node that isn't local to the head node is never checked -- no collision is possible."""
        daq_config = DaqConfig(
            head_node_data_dir="/data/head",
            head_node_ip_addr="10.0.0.1",
            head_node_container=False,
            daq_nodes=[
                DaqNode(username="panoseti", data_dir="/data", ip_addr="192.168.0.10", module_ids=[0])
            ],
        )
        _patch_configs(monkeypatch, daq_config, NetworkConfig())

        results = _check_port_collision()

        assert results == []
