import os
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from control.admin.cli import app

runner = CliRunner()

def get_mock_env():
    env = os.environ.copy()
    env["HEADNODE_IP"] = "127.0.0.1"
    return env


class _FakeStdout:
    """Async-iterable stand-in for a subprocess's stdout pipe -- empty (no output lines)."""
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeProcess:
    """Stand-in for asyncio.create_subprocess_exec()'s return value: succeeds immediately, no output."""
    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.stdout = _FakeStdout()

    async def wait(self):
        return self.returncode


def _mock_create_subprocess_exec():
    """build/deploy now run compose via run_cmd_async() (asyncio.create_subprocess_exec),
    not subprocess.run -- see run_cmd_async()'s docstring for why (concurrent multi-node
    output would garble via N subprocess.run()s sharing one terminal fd). subprocess.run is
    still used (via asyncio.to_thread) for the `docker context ls` existence check in
    deploy_node(), so that mock is unaffected and stays separate.
    """
    return patch('control.admin.cli.asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=_FakeProcess())


@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
def test_admin_build_headnode():
    with _mock_create_subprocess_exec() as mock_exec:
        result = runner.invoke(app, ["build", "headnode"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"

        # We expect a docker compose build call for headnode
        found_headnode_build = False
        for call_args in mock_exec.call_args_list:
            cmd = call_args[0]
            if "docker" in cmd and "compose" in cmd and "-p" in cmd and "pseti-headnode" in cmd and "build" in cmd:
                found_headnode_build = True

        assert found_headnode_build, f"Expected headnode build command, got: {mock_exec.call_args_list}"

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
@patch('control.admin.cli.get_docker_context_for_node', return_value='panoseti-192.168.1.10')
@patch('control.utils.util.is_local', return_value=False)
@patch('control.utils.config_file.get_daq_config')
def test_admin_deploy_node(mock_get_daq_config, mock_is_local, mock_get_ctx):
    with patch('control.admin.cli.subprocess.run') as mock_run, _mock_create_subprocess_exec() as mock_exec:
        # mock docker context ls (still a subprocess.run call, via asyncio.to_thread)
        # to return our context
        mock_run.return_value.stdout = "panoseti-192.168.1.10"

        result = runner.invoke(app, ["deploy", "192.168.1.10"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"

        found_daqnode = False
        found_alloy = False
        for call_args in mock_exec.call_args_list:
            cmd = call_args[0]
            if "docker" in cmd and "--context" in cmd and "pseti-daqnode-192-168-1-10" in cmd:
                if "docker-compose.daqnode.yml" in str(cmd):
                    found_daqnode = True
                elif "docker-compose.alloy.yml" in str(cmd):
                    found_alloy = True

        assert found_daqnode
        assert found_alloy

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
@patch('control.admin.cli.get_docker_context_for_node', return_value='panoseti-192.168.1.10')
@patch('control.utils.util.is_local', return_value=False)
@patch('control.utils.config_file.get_daq_config')
def test_admin_down_all(mock_get_daq_config, mock_is_local, mock_get_ctx):
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

@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1", "PSETI_HEADNODE_DISABLE_SERVICES": "redis,influxdb"})
def test_admin_deploy_headnode_disable_services():
    """PSETI_HEADNODE_DISABLE_SERVICES should omit named services from the
    compose `up` command's service args while still including the
    never-optional ones (loki, alloy, headnode-server) and any optional
    service *not* named (grafana)."""
    with _mock_create_subprocess_exec() as mock_exec:
        result = runner.invoke(app, ["deploy", "headnode"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"

        found = False
        for call_args in mock_exec.call_args_list:
            cmd = call_args[0]
            if "up" in cmd and "pseti-headnode" in cmd:
                found = True
                assert "redis" not in cmd, f"redis should be omitted: {cmd}"
                assert "influxdb" not in cmd, f"influxdb should be omitted: {cmd}"
                for expected in ("grafana", "loki", "alloy", "headnode-server"):
                    assert expected in cmd, f"{expected} should still be present: {cmd}"
        assert found, f"Expected headnode up command, got: {mock_exec.call_args_list}"


@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
@patch('control.admin.cli._grpc_pinned_commit', return_value='abc1234')
@patch('control.admin.cli._resolve_bare_metal_ssh_target', return_value=['192.168.0.228'])
def test_admin_deploy_bare_metal_dry_run(mock_ssh_target, mock_pinned_commit):
    """--dry-run must print the per-node bare-metal commands (env file write,
    pinned pip install, systemctl restarts) and must NOT touch the network --
    no `ssh`/`sudo` subprocess, no docker compose call either. This is the
    replacement for a real bare-metal deploy when an operator doesn't want
    `pseti admin` running `sudo` over non-interactive SSH on their behalf
    (see deploy_node()'s bare-metal branch: the old hardcoded
    `echo <password> | sudo -S` is gone, and --dry-run is the alternative
    for a site without passwordless sudo configured)."""
    with patch('control.admin.cli.subprocess.run') as mock_run, _mock_create_subprocess_exec() as mock_exec:
        result = runner.invoke(app, ["deploy", "192.168.0.228", "--mode", "bare-metal", "--dry-run"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"

        # No SSH/subprocess execution of any kind -- this is a print-only path.
        mock_run.assert_not_called()
        mock_exec.assert_not_called()

        assert "abc1234" in result.stdout
        assert "pip install --upgrade" in result.stdout
        assert "sudo systemctl restart panoseti_grpc" in result.stdout
        assert "sudo systemctl restart panoseti_alloy" in result.stdout
        # The literal hardcoded password must never appear.
        assert "echo panoseti" not in result.stdout


@patch.dict(os.environ, {"HEADNODE_IP": "127.0.0.1"})
def test_admin_attach():
    with patch('control.admin.cli.subprocess.run') as mock_run:
        result = runner.invoke(app, ["attach", "headnode"])
        assert result.exit_code == 0, f"Command failed: {result.stdout}"
        
        found_headnode_exec = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if "exec" in cmd and "-it" in cmd and "headnode-server" in cmd and "pseti-headnode" in cmd:
                found_headnode_exec = True
        
        assert found_headnode_exec
