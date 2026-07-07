import os
from unittest.mock import patch, call
from typer.testing import CliRunner

from control.admin.cli import app

runner = CliRunner()

def get_mock_env():
    env = os.environ.copy()
    env["HEADNODE_IP"] = "127.0.0.1"
    return env

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
def test_admin_build_headnode():
    with patch('control.admin.cli.subprocess.run') as mock_run:
        result = runner.invoke(app, ["build", "headnode"])
        assert result.exit_code == 0
        
        # We expect a docker compose build call for headnode
        found_headnode_build = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if "docker" in cmd and "compose" in cmd and "-p" in cmd and "pseti-headnode" in cmd and "build" in cmd:
                found_headnode_build = True
        
        assert found_headnode_build, f"Expected headnode build command, got: {mock_run.call_args_list}"

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
@patch('control.admin.cli.get_docker_context_for_node', return_value='panoseti-192.168.1.10')
def test_admin_deploy_node(mock_get_ctx):
    with patch('control.admin.cli.subprocess.run') as mock_run:
        # mock docker context ls to return our context
        mock_run.return_value.stdout = "panoseti-192.168.1.10"
        
        result = runner.invoke(app, ["deploy", "192.168.1.10"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"
        
        found_daqnode = False
        found_alloy = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if "docker" in cmd and "--context" in cmd and "pseti-daqnode-192-168-1-10" in cmd:
                if "docker-compose.daqnode.yml" in str(cmd):
                    found_daqnode = True
                elif "docker-compose.alloy.yml" in str(cmd):
                    found_alloy = True
                    
        assert found_daqnode
        assert found_alloy

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
@patch('control.admin.cli.get_docker_context_for_node', return_value='panoseti-192.168.1.10')
def test_admin_down_all(mock_get_ctx):
    with patch('control.admin.cli.subprocess.run') as mock_run:
        # mock docker context ls to return our context
        mock_run.return_value.stdout = "panoseti-192.168.1.10"
        
        result = runner.invoke(app, ["down", "headnode,192.168.1.10"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"
        
        found_headnode = False
        found_daqnode = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if "docker" in cmd and "down" in cmd:
                if "pseti-headnode" in cmd:
                    found_headnode = True
                elif "pseti-daqnode-192-168-1-10" in cmd:
                    found_daqnode = True
                    
        assert found_headnode
        assert found_daqnode

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
def test_admin_attach():
    with patch('control.admin.cli.subprocess.run') as mock_run:
        result = runner.invoke(app, ["attach", "headnode"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"
        
        found_headnode_logs = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if "logs" in cmd and "-f" in cmd and "daqnode-server" in cmd and "pseti-headnode" in cmd:
                found_headnode_logs = True
        
        assert found_headnode_logs
