"""
test_grpc_endpoint.py

Unit tests for control.utils.util.resolve_grpc_port() and daq_grpc_endpoint()
-- the client-side half of the single source of truth for gRPC ports (the
server-side half is panoseti_grpc.unified_main.resolve_bind_port(), tested
in the grpc/ submodule).

Root cause under test: before this, the unified server's bind port and the
control-plane client's connect port were resolved by four independent
mechanisms that could silently desync (a hardcoded 50051 in
daq_grpc_endpoint(), an unread GRPC_PORT default_factory, a headnode
profile that shipped with `port = 50052` baked into its TOML, ...). These
tests pin the precedence that replaces all of that: explicit per-node
config > role env var > legacy env var > 50051 default -- the exact chain
resolve_bind_port() applies on the server side.
"""
from __future__ import annotations

import pytest

from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    NetworkConfig,
    NetworkDaqNode,
    PortForwarding,
    TransferNodeSpec,
)
from control.utils.util import attach_daq_config, daq_grpc_endpoint, resolve_grpc_port

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PORT_ENV_VARS = ("HEADNODE_GRPC_PORT", "DAQNODE_GRPC_PORT", "DAQ_DATA_GATEWAY_PORT", "GRPC_PORT")


@pytest.fixture(autouse=True)
def _clean_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with none of the resolver's env vars set.

    Without this, a var leaking from the developer's shell or a prior test
    would silently change which precedence branch each assertion actually
    exercises -- exactly the kind of nondeterminism this resolver exists to
    eliminate at the *server* end; the tests shouldn't reintroduce it here.
    """
    for var in _PORT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _daq_node(**overrides: object) -> DaqNode:
    defaults: dict = dict(
        username="panoseti",
        data_dir="/data",
        ip_addr="192.168.0.228",
        module_ids=[254],
    )
    defaults.update(overrides)
    return DaqNode(**defaults)


# ---------------------------------------------------------------------------
# resolve_grpc_port — precedence matrix
# ---------------------------------------------------------------------------


class TestResolveGrpcPortPrecedence:
    def test_default_when_nothing_set(self) -> None:
        assert resolve_grpc_port("headnode") == 50051
        assert resolve_grpc_port("daqnode") == 50051

    def test_legacy_grpc_port_is_lowest_priority_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRPC_PORT", "40000")
        assert resolve_grpc_port("headnode") == 40000
        assert resolve_grpc_port("daqnode") == 40000

    def test_deprecated_daq_data_gateway_port_alias_for_headnode_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAQ_DATA_GATEWAY_PORT", "41000")
        assert resolve_grpc_port("headnode") == 41000
        # daqnode has no legacy alias to DAQ_DATA_GATEWAY_PORT -- falls all
        # the way through to the built-in default instead.
        assert resolve_grpc_port("daqnode") == 50051

    def test_canonical_role_var_wins_over_deprecated_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAQ_DATA_GATEWAY_PORT", "41000")
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50052")
        assert resolve_grpc_port("headnode") == 50052

    def test_canonical_role_var_wins_over_legacy_grpc_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRPC_PORT", "40000")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50099")
        assert resolve_grpc_port("daqnode") == 50099

    def test_explicit_wins_over_every_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50052")
        monkeypatch.setenv("GRPC_PORT", "40000")
        assert resolve_grpc_port("headnode", explicit=9999) == 9999

    def test_head_and_daq_roles_resolve_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The entire reason two vars exist: a co-located node needs them to differ."""
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50051")
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50052")
        assert resolve_grpc_port("headnode") != resolve_grpc_port("daqnode")

    def test_unknown_role_has_no_env_vars_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADNODE_GRPC_PORT", "50052")
        monkeypatch.setenv("GRPC_PORT", "40000")
        # An unrecognized role skips straight to the legacy var, then default.
        assert resolve_grpc_port("bogus-role") == 40000


# ---------------------------------------------------------------------------
# daq_grpc_endpoint — the three real branches
# ---------------------------------------------------------------------------


class TestDaqGrpcEndpoint:
    def test_local_node_uses_daqnode_role_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch 1 (is_local): must resolve via the daqnode role, not a hardcoded 50051."""
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50077")
        node = _daq_node(ip_addr="192.168.0.228")

        from control.utils import util as util_mod

        monkeypatch.setattr(util_mod, "is_local", lambda ip, cfg: True)
        host, port = daq_grpc_endpoint(node, daq_config=object())
        assert (host, port) == ("192.168.0.228", 50077)

    def test_forwarded_node_with_explicit_grpc_port_uses_gateway(self) -> None:
        """Branch 2: an operator-set port_forwarding.grpc_port routes through the gateway IP."""
        node = _daq_node(
            ip_addr="192.168.0.228",
            port_forwarding=PortForwarding(status=True, gw_ip="192.168.88.152", grpc_port=50051),
        )
        assert daq_grpc_endpoint(node, daq_config=None) == ("192.168.88.152", 50051)

    def test_forwarded_node_without_explicit_grpc_port_falls_through_to_direct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """grpc_port defaulting to None (not 50051) is what makes this fallthrough possible.

        Before PortForwarding.grpc_port's default changed from 50051 to
        None, "operator forwarded gRPC explicitly" and "operator didn't set
        grpc_port at all" were indistinguishable, so this branch could never
        be taken for a forwarded-but-not-gRPC-forwarded node.
        """
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50088")
        node = _daq_node(
            ip_addr="192.168.0.228",
            port_forwarding=PortForwarding(status=True, gw_ip="192.168.88.152"),
        )
        assert daq_grpc_endpoint(node, daq_config=None) == ("192.168.0.228", 50088)

    def test_direct_node_uses_env_resolved_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch 3: no port_forwarding at all -- direct IP, env-resolved port."""
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50099")
        node = _daq_node(ip_addr="10.0.0.5")
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.0.5", 50099)

    def test_direct_node_with_own_grpc_port_override_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A per-node DaqNode.grpc_port override beats the fleet-wide env var."""
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50099")
        node = _daq_node(ip_addr="10.0.0.6", grpc_port=60000)
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.0.6", 60000)

    def test_no_env_no_override_falls_back_to_50051(self) -> None:
        node = _daq_node(ip_addr="10.0.0.7")
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.0.7", 50051)


# ---------------------------------------------------------------------------
# PortForwarding.grpc_port default -- the backward-compat-sensitive change
# ---------------------------------------------------------------------------


class TestPortForwardingGrpcPortDefault:
    def test_omitted_grpc_port_is_none_not_50051(self) -> None:
        """Regression guard for the exact bug daq_grpc_endpoint's branch-2 fallthrough depends on."""
        pf = PortForwarding(status=True, gw_ip="10.0.1.254")
        assert pf.grpc_port is None

    def test_explicit_grpc_port_still_respected(self) -> None:
        pf = PortForwarding(status=True, gw_ip="10.0.1.254", grpc_port=12345)
        assert pf.grpc_port == 12345


# ---------------------------------------------------------------------------
# attach_daq_config() -- network_config.json's direct-connect grpc_port as
# a new middle precedence tier between daq_config.json's own explicit
# override and resolve_grpc_port()'s env var / 50051 default.
# ---------------------------------------------------------------------------


def _network_config(**node_overrides: object) -> NetworkConfig:
    defaults: dict = dict(
        ip_addr="192.168.0.228",
        port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
    )
    defaults.update(node_overrides)
    return NetworkConfig(daq_nodes=[NetworkDaqNode(**defaults)])  # type: ignore


class TestAttachDaqConfigGrpcPort:
    def test_network_config_grpc_port_fills_in_when_daq_node_unset(self) -> None:
        daq_config = DaqConfig(head_node_data_dir="/data/head", head_node_ip_addr="10.99.99.99", daq_nodes=[_daq_node(ip_addr="192.168.0.228")])
        network_config = _network_config(ip_addr="192.168.0.228", grpc_port=50077)

        attach_daq_config(daq_config, network_config)

        assert daq_config.daq_nodes[0].grpc_port == 50077

    def test_daq_config_own_explicit_grpc_port_wins_over_network_config(self) -> None:
        daq_config = DaqConfig(
            head_node_data_dir="/data/head",
            head_node_ip_addr="10.99.99.99",
            daq_nodes=[_daq_node(ip_addr="192.168.0.228", grpc_port=50099)],
        )
        network_config = _network_config(ip_addr="192.168.0.228", grpc_port=50077)

        attach_daq_config(daq_config, network_config)

        assert daq_config.daq_nodes[0].grpc_port == 50099

    def test_neither_set_stays_none(self) -> None:
        daq_config = DaqConfig(head_node_data_dir="/data/head", head_node_ip_addr="10.99.99.99", daq_nodes=[_daq_node(ip_addr="192.168.0.228")])
        network_config = _network_config(ip_addr="192.168.0.228")

        attach_daq_config(daq_config, network_config)

        assert daq_config.daq_nodes[0].grpc_port is None

    def test_no_matching_network_config_node_leaves_grpc_port_unset(self) -> None:
        daq_config = DaqConfig(head_node_data_dir="/data/head", head_node_ip_addr="10.99.99.99", daq_nodes=[_daq_node(ip_addr="192.168.0.228")])
        network_config = _network_config(ip_addr="10.0.0.99", grpc_port=50077)

        attach_daq_config(daq_config, network_config)

        assert daq_config.daq_nodes[0].grpc_port is None

    def test_end_to_end_daq_grpc_endpoint_resolves_network_config_port(self) -> None:
        """The full chain: network_config.json's sibling grpc_port reaches
        daq_grpc_endpoint()'s resolved (host, port) via attach_daq_config(),
        with no changes needed to daq_grpc_endpoint() itself.
        """
        daq_config = DaqConfig(head_node_data_dir="/data/head", head_node_ip_addr="10.99.99.99", daq_nodes=[_daq_node(ip_addr="192.168.0.228")])
        network_config = _network_config(ip_addr="192.168.0.228", grpc_port=50077)

        attach_daq_config(daq_config, network_config)
        node = daq_config.daq_nodes[0]

        assert daq_grpc_endpoint(node, daq_config) == ("192.168.0.228", 50077)


# ---------------------------------------------------------------------------
# TransferNodeSpec -- the Transfer Daemon's node type. It never has a
# daq_config to pass to daq_grpc_endpoint() (it only ever sees the snapshot
# in the job TOML), so its own grpc_port override is the only way it can
# reach a non-default direct-connection port.
# ---------------------------------------------------------------------------


def _transfer_node(**overrides: object) -> TransferNodeSpec:
    defaults: dict = dict(
        username="panoseti",
        data_dir="/data",
        ip_addr="192.168.0.228",
        module_ids=[254],
    )
    defaults.update(overrides)
    return TransferNodeSpec(**defaults)


class TestDaqGrpcEndpointWithTransferNodeSpec:
    def test_direct_node_with_own_grpc_port_override_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TransferNodeSpec.grpc_port must be honored the same way DaqNode.grpc_port is --
        this is the only way the Transfer Daemon (which never has a daq_config to pass to
        daq_grpc_endpoint()) can resolve a per-node port override."""
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50099")
        node = _transfer_node(ip_addr="10.0.0.6", grpc_port=60000)
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.0.6", 60000)

    def test_direct_node_without_override_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAQNODE_GRPC_PORT", "50099")
        node = _transfer_node(ip_addr="10.0.0.7")
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.0.7", 50099)

    def test_forwarded_node_with_explicit_grpc_port_uses_gateway(self) -> None:
        node = _transfer_node(
            ip_addr="10.0.0.8",
            port_forwarding=PortForwarding(status=True, gw_ip="10.0.1.254", grpc_port=12345),
        )
        assert daq_grpc_endpoint(node, daq_config=None) == ("10.0.1.254", 12345)
