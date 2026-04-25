"""
test_pydantic_models_transfer.py

Unit tests for TransferNodeSpec and TransferJob Pydantic models.
"""

from datetime import datetime, timezone

import pytest

from control.utils.pydantic_config_models import (
    PortForwarding,
    TransferJob,
    TransferNodeSpec,
)


class TestTransferNodeSpec:
    """Test cases for TransferNodeSpec model."""

    def test_valid_node_spec_without_port_forwarding(self):
        """TransferNodeSpec validates correctly with port_forwarding=None."""
        spec = TransferNodeSpec(
            ip_addr="192.168.0.10",
            username="daq_user",
            data_dir="/data/runs",
            module_ids=[0, 1, 2, 3],
            port_forwarding=None,
        )
        assert str(spec.ip_addr) == "192.168.0.10"
        assert spec.username == "daq_user"
        assert spec.data_dir == "/data/runs"
        assert spec.module_ids == [0, 1, 2, 3]
        assert spec.port_forwarding is None

    def test_valid_node_spec_with_port_forwarding(self):
        """TransferNodeSpec validates correctly with a full PortForwarding object."""
        pf = PortForwarding(
            status=True,
            gw_ip="10.0.1.254",
            reboot_port=[60000, 60001, 60002, 60003],
            cmd_port=[60004, 60005, 60006, 60007],
            grpc_port=50051,
        )
        spec = TransferNodeSpec(
            ip_addr="192.168.0.20",
            username="daq_user",
            data_dir="/data/runs",
            module_ids=[4, 5],
            port_forwarding=pf,
        )
        assert spec.port_forwarding is not None
        assert spec.port_forwarding.status is True
        assert str(spec.port_forwarding.gw_ip) == "10.0.1.254"
        assert spec.port_forwarding.grpc_port == 50051

    def test_node_spec_rejects_extra_fields(self):
        """TransferNodeSpec rejects extra fields (BaseStrictModel)."""
        with pytest.raises(Exception):  # Pydantic validation error
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[0, 1],
                port_forwarding=None,
                extra_field="should_fail",
            )


class TestTransferJob:
    """Test cases for TransferJob model."""

    def test_valid_transfer_job_with_multiple_nodes(self):
        """TransferJob validates with a list of TransferNodeSpec."""
        now = datetime.now(timezone.utc)
        nodes = [
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[0, 1, 2, 3],
            ),
            TransferNodeSpec(
                ip_addr="192.168.0.20",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[4, 5, 6, 7],
            ),
        ]
        job = TransferJob(
            schema_version=1,
            run_name="run_2024_04_01_120000",
            head_data_dir="/data/runs",
            head_node_username="control_user",
            created_at=now,
            attempts=0,
            daq_nodes=nodes,
        )
        assert job.run_name == "run_2024_04_01_120000"
        assert len(job.daq_nodes) == 2
        assert job.schema_version == 1
        assert job.attempts == 0

    def test_port_forwarding_round_trip(self):
        """Port-forwarding round-trip: serialize and deserialize preserves data."""
        now = datetime.now(timezone.utc)
        pf = PortForwarding(
            status=True,
            gw_ip="10.0.1.254",
            reboot_port=[60000, 60001, 60002, 60003],
            cmd_port=[60004, 60005, 60006, 60007],
            grpc_port=50051,
        )
        nodes = [
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[0, 1, 2, 3],
                port_forwarding=pf,
            ),
        ]
        job = TransferJob(
            schema_version=1,
            run_name="run_with_pf",
            head_data_dir="/data/runs",
            head_node_username="control_user",
            created_at=now,
            daq_nodes=nodes,
        )

        dumped = job.model_dump()
        restored = TransferJob.model_validate(dumped)

        assert restored.daq_nodes[0].port_forwarding is not None
        assert str(restored.daq_nodes[0].port_forwarding.gw_ip) == "10.0.1.254"
        assert restored.daq_nodes[0].port_forwarding.grpc_port == 50051

    def test_transfer_job_rejects_extra_fields(self):
        """TransferJob rejects extra fields (BaseStrictModel)."""
        now = datetime.now(timezone.utc)
        nodes = [
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[0, 1],
            ),
        ]
        with pytest.raises(Exception):  # Pydantic validation error
            TransferJob(
                schema_version=1,
                run_name="test_run",
                head_data_dir="/data/runs",
                head_node_username="control_user",
                created_at=now,
                daq_nodes=nodes,
                extra_field="should_fail",
            )

    def test_transfer_job_defaults(self):
        """TransferJob uses correct default values."""
        now = datetime.now(timezone.utc)
        nodes = [
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="daq_user",
                data_dir="/data/runs",
                module_ids=[0, 1],
            ),
        ]
        job = TransferJob(
            run_name="test_run",
            head_data_dir="/data/runs",
            head_node_username="control_user",
            created_at=now,
            daq_nodes=nodes,
        )
        assert job.schema_version == 1
        assert job.attempts == 0
        assert job.no_cleanup is False
        assert job.no_collect is False
        assert job.skip_verify is False
