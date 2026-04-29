"""
test_grpc_cli.py — Integration tests for the pseti grpc command.
Verifies that the CLI correctly routes to and interacts with gRPC services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from control.pseti import app

runner = CliRunner()


@pytest.fixture
def mock_grpc_channels():
    """Mock the gRPC channel and stubs to avoid actual network calls."""
    with patch("panoseti_grpc._cli.root._make_channel") as mock_ch, \
         patch("panoseti_grpc._cli.daq_data._make_channel") as mock_data_ch, \
         patch("panoseti_grpc._cli.daq_control._make_channel") as mock_ctrl_ch:
        yield mock_ch, mock_data_ch, mock_ctrl_ch


def test_pseti_grpc_help():
    """Verify that the pseti grpc subcommand displays help correctly."""
    result = runner.invoke(app, ["grpc", "--help"])
    assert result.exit_code == 0
    assert "PSETI unified gRPC CLI." in result.output, f"{result.output=}"
    assert "stat" in result.output
    assert "telemetry" in result.output


def test_pseti_grpc_stat_success(mock_grpc_channels):
    """Verify pseti grpc stat reports OK when services are responsive."""
    _, _, _ = mock_grpc_channels
    
    # Mock DaqControl StatusDaq response
    mock_ctrl_resp = MagicMock()
    mock_ctrl_resp.success = True
    mock_ctrl_resp.hashpipe_running = True
    
    # Mock Telemetry Log response
    mock_telem_resp = MagicMock()
    mock_telem_resp.success = True
    
    # Patch stubs and client where they are USED in root.py
    with patch("panoseti_grpc._cli.root.daq_data_pb2_grpc.DaqDataStub"), \
         patch("panoseti_grpc._cli.root.daq_control_pb2_grpc.DaqControlStub") as mock_ctrl_stub, \
         patch("panoseti_grpc._cli.root.TelemetryClient") as mock_telem_client:
         
        # Setup mocks
        mock_ctrl_stub.return_value.StatusDaq.return_value = mock_ctrl_resp
        mock_telem_client.return_value.send_log_future.return_value.result.return_value = mock_telem_resp
        
        result = runner.invoke(app, ["grpc", "stat"])
        if result.exit_code != 0:
            print(f"EXIT CODE: {result.exit_code}")
            print(f"OUTPUT: {result.output}")
            if result.exception:
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

        assert result.exit_code == 0
        assert "daq_data" in result.output
        assert "daq_control" in result.output
        assert "telemetry" in result.output
        assert "✓ OK" in result.output


def test_pseti_grpc_daq_data_ping(mock_grpc_channels):
    """Verify pseti grpc daq-data ping command."""
    _, _mock_data_ch, _ = mock_grpc_channels
    
    with patch("panoseti_grpc._cli.daq_data.daq_data_pb2_grpc.DaqDataStub") as mock_stub:
        result = runner.invoke(app, ["grpc", "daq-data", "ping"])
        assert result.exit_code == 0
        assert "DaqData Ping OK" in result.output
        mock_stub.return_value.Ping.assert_called_once()


def test_pseti_grpc_daq_control_stat(mock_grpc_channels):
    """Verify pseti grpc daq-control stat command with flags."""
    _, _, _mock_ctrl_ch = mock_grpc_channels
    
    mock_resp = MagicMock()
    mock_resp.success = True
    mock_resp.hashpipe_running = True
    
    with patch("panoseti_grpc._cli.daq_control.daq_control_pb2_grpc.DaqControlStub") as mock_stub:
        mock_stub.return_value.StatusDaq.return_value = mock_resp
        result = runner.invoke(app, ["grpc", "daq-control", "stat", "--no-disk", "--no-runs"])
        assert result.exit_code == 0
        assert "Hashpipe running" in result.output
        # Verify requested flags in proto call
        call_args = mock_stub.return_value.StatusDaq.call_args[0][0]
        assert call_args.check_hashpipe_running is True
        assert call_args.check_disk_usage is False
        assert call_args.check_run_dirs is False


def test_pseti_grpc_daq_control_get_manifest(mock_grpc_channels):
    """Verify pseti grpc daq-control get-manifest command."""
    _, _, _mock_ctrl_ch = mock_grpc_channels
    
    # Mock streaming response
    mock_entry = MagicMock()
    mock_entry.relative_path = "test.pff"
    mock_entry.digest_hex = "abcdef"
    mock_entry.size_bytes = 1234
    
    with patch("panoseti_grpc._cli.daq_control.daq_control_pb2_grpc.DaqControlStub") as mock_stub:
        mock_stub.return_value.GetManifest.return_value = [mock_entry]
        result = runner.invoke(app, ["grpc", "daq-control", "get-manifest", "--run-dir", "run01", "--module-id", "200"])
        if result.exit_code != 0:
            print(f"OUTPUT: {result.output}")
            if result.exception:
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        assert result.exit_code == 0
        assert "test.pff" in result.output
        assert "abcdef" in result.output


def test_pseti_grpc_telemetry_test(mock_grpc_channels):
    """Verify pseti grpc telemetry test command."""
    with patch("panoseti_grpc._cli.telemetry.TelemetryClient") as mock_client:
        result = runner.invoke(app, ["grpc", "telemetry", "test", "--count", "2"])
        assert result.exit_code == 0
        assert "Sending 2 test logs" in result.output
        assert mock_client.return_value.log_flexible.call_count == 2


def test_pseti_grpc_server_help():
    """Verify pseti grpc server subcommand displays help correctly."""
    result = runner.invoke(app, ["grpc", "server", "--help"])
    assert result.exit_code == 0
    assert "Manage and run the unified gRPC server." in result.output
    assert "--profile" in result.output
    assert "--config" in result.output


# def test_pseti_grpc_test_help():
#     """Verify pseti grpc test subcommand displays help correctly."""
#     result = runner.invoke(app, ["grpc", "test", "--help"])
#     assert result.exit_code == 0
#     assert "PSETI Unified QA Runner" in result.output
