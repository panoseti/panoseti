"""
synth.py — FleetSpec → Topology.

Translates a FleetSpec into validated Pydantic config models, runs
GlobalConfigValidator, builds the topology graph, and returns a Topology.
"""

from __future__ import annotations

import os
from ipaddress import IPv4Address

from control.topology.fleet import (
    generate_daemons_config,
    generate_data_config,
    generate_firmware_config,
)
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    FirmwareConfig,
    NetworkConfig,
    NetworkDaqNode,
    NetworkModule,
    ObsConfig,
    ObsDomeConfig,
    ObsModuleConfig,
    PortForwarding,
    QuaboUidDome,
    QuaboUidEntry,
    QuaboUidModule,
    QuaboUids,
)

from ci.software_only_v2.infra.spec import FleetSpec, Topology


def realize(spec: FleetSpec) -> Topology:
    """
    Translate a FleetSpec into a fully-validated Topology.

    Raises:
        ValueError: If GlobalConfigValidator reports any ERROR-level issues.
    """
    import copy

    obs = _build_obs_config(spec)
    daq = _build_daq_config(spec)
    network = _build_network_config(spec)
    quabo_uids = _build_quabo_uids(spec)
    data = generate_data_config(
        run_type=spec.data_spec.run_type,
        overvoltage=spec.data_spec.overvoltage,
        integration_time_usec=spec.data_spec.integration_time_usec,
        pe_threshold=spec.data_spec.pe_threshold,
        quabo_sample_size=spec.data_spec.quabo_sample_size,
    )
    firmware = generate_firmware_config(
        qfp=spec.firmware_spec.qfp,
        bga=spec.firmware_spec.bga,
    )
    daemons = generate_daemons_config()

    # Use deep copies for graph building and validation — both operations call
    # config_file.associate(), which injects circular back-references
    # (DaqNode.modules → QuaboUidModule → .daq_node → DaqNode).  Keeping the
    # originals clean lets materialize.py serialize them without hitting
    # PydanticSerializationError: Circular reference detected.
    daq_for_ops = copy.deepcopy(daq)
    quabo_uids_for_ops = copy.deepcopy(quabo_uids)

    from control.topology.graph_builder import GraphBuilder
    graph = GraphBuilder().build_from_configs(daq_for_ops, quabo_uids_for_ops, obs, network)

    # Run GlobalConfigValidator — skip firmware filesystem check in tests
    # by passing firmware_config=None (binary files don't exist in test env).
    _run_validator(obs, data, daq_for_ops, network, quabo_uids_for_ops)

    return Topology(
        obs=obs,
        daq=daq,
        network=network,
        data=data,
        firmware=firmware,
        quabo_uids=quabo_uids,
        daemons=daemons,
        graph=graph,
        name=spec.name,
    )


def _build_obs_config(spec: FleetSpec) -> ObsConfig:
    domes = []
    for di, dome_spec in enumerate(spec.domes):
        modules = []
        for mod in dome_spec.modules:
            modules.append(ObsModuleConfig(
                mobo_serialno=mod.mobo_serialno or f"M{mod.module_id:03d}",
                quabo_version=mod.version,
                ip_addr=IPv4Address(mod.ip),
                timing_mode=mod.timing,
                wps=mod.wps,
                id=mod.module_id,
            ))
        domes.append(ObsDomeConfig(
            name=dome_spec.name,
            obslat=dome_spec.lat,
            obslon=dome_spec.lon,
            obsalt=dome_spec.alt,
            modules=modules,
            num=di,
        ))
    obs = ObsConfig(
        name=spec.name,
        domes=domes,
        detector_overvoltage=spec.data_spec.overvoltage,  # type: ignore[arg-type]
    )
    # Inject a default WPS definition so _check_wps_references passes when
    # any module references "wps" (the default value).
    obs.wps = {"url": "http://192.168.1.1", "quabo_socket": 4}  # type: ignore[assignment]
    return obs


def _build_daq_config(spec: FleetSpec) -> DaqConfig:
    head_data_dir = os.environ.get("HEAD_DATA_DIR", spec.head_data_dir)
    nodes = []
    for node_spec in spec.daq_nodes:
        pf: PortForwarding | None = None
        if node_spec.gateway:
            gw = node_spec.gateway
            pf = PortForwarding(
                status=True,
                gw_ip=IPv4Address(gw.ip),
                port=gw.ssh_port,
                grpc_port=gw.grpc_port,
            )
        nodes.append(DaqNode(
            username=node_spec.username,
            data_dir=node_spec.data_dir,
            ip_addr=IPv4Address(node_spec.ip),
            module_ids=node_spec.module_ids,
            bindhost=node_spec.bindhost,
            port_forwarding=pf,
        ))
    # head_node_container=True bypasses strict IP-reachability checks in CI
    return DaqConfig(
        head_node_data_dir=head_data_dir,
        head_node_ip_addr=IPv4Address(spec.headnode_ip),
        head_node_container=True,
        daq_nodes=nodes,
    )


def _build_network_config(spec: FleetSpec) -> NetworkConfig:
    net_modules: list[NetworkModule] = []
    net_daq_nodes: list[NetworkDaqNode] = []
    for node_spec in spec.daq_nodes:
        if node_spec.gateway:
            gw = node_spec.gateway
            gw_ip = IPv4Address(gw.ip)
            pf = PortForwarding(
                status=True,
                gw_ip=gw_ip,
                port=gw.ssh_port,
                grpc_port=gw.grpc_port,
            )
            net_daq_nodes.append(NetworkDaqNode(
                ip_addr=IPv4Address(node_spec.ip),
                port_forwarding=pf,
            ))
            # Add network PF entries for each module on this gateway node
            for mid in node_spec.module_ids:
                mod_ip = _module_id_to_ip(mid, spec)
                if mod_ip:
                    net_modules.append(NetworkModule(
                        ip_addr=IPv4Address(mod_ip),
                        port_forwarding=PortForwarding(
                            status=True,
                            gw_ip=gw_ip,
                            reboot_port=[69, 60004, 60005, 60006],
                            cmd_port=[60000, 60001, 60002, 60003],
                        ),
                    ))
    return NetworkConfig(modules=net_modules, daq_nodes=net_daq_nodes)


def _build_quabo_uids(spec: FleetSpec) -> QuaboUids:
    all_modules: list[QuaboUidModule] = []
    for dome_spec in spec.domes:
        for mod in dome_spec.modules:
            all_modules.append(QuaboUidModule(
                ip_addr=IPv4Address(mod.ip),
                quabos=[QuaboUidEntry(uid=f"q_{mod.module_id}_{k}") for k in range(4)],
                id=mod.module_id,
            ))
    return QuaboUids(domes=[QuaboUidDome(num=0, modules=all_modules)])


def _module_id_to_ip(module_id: int, spec: FleetSpec) -> str | None:
    for dome in spec.domes:
        for mod in dome.modules:
            if mod.module_id == module_id:
                return mod.ip
    return None


def _run_validator(
    obs: ObsConfig,
    data: object,
    daq: DaqConfig,
    network: NetworkConfig,
    quabo_uids: QuaboUids,
) -> None:
    from control.utils.global_validator import validate_all
    # firmware_config=None skips the firmware-filesystem check (_check_firmware_filesystem)
    # since test environments don't have real firmware binary files on disk.
    # validate_all() raises ValueError if any _check_* rule reports ERROR.
    validate_all(
        obs_config=obs,
        data_config=data,  # type: ignore[arg-type]
        daq_config=daq,
        network_config=network,
        firmware_config=None,
        quabo_uids=quabo_uids,
    )
