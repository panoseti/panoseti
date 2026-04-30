"""
Tests for the proposed UDP-based util.ping function.

These tests verify the behavior of a safer, non-ICMP reachability check
that works reliably across port-forwarded gateway environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# Proposed implementation for testing
def proposed_udp_ping(ip_addr: str, cmd_port: int) -> bool:
    from control.driver.quabo_driver import QUABO
    
    q = QUABO(ip_addr, cmd_port)
    try:
        cmd = q.make_cmd(0x01)
        q.send(cmd)
        q.sock.settimeout(1.0)
        data, _ = q.sock.recvfrom(1024)
        return bool(data)
    except (TimeoutError, OSError):
        return False
    finally:
        q.close()


@patch("control.driver.quabo_driver.QUABO")
def test_proposed_udp_ping_success(mock_quabo_cls) -> None:
    """If the socket receives data back, ping is True."""
    mock_q = MagicMock()
    mock_quabo_cls.return_value = mock_q
    
    # Mock make_cmd
    mock_q.make_cmd.return_value = bytearray(64)
    mock_q.make_cmd.return_value[0] = 0x01
    
    # Mock recvfrom to return a simulated HK/ACK payload
    mock_q.sock.recvfrom.return_value = (b"\x01\x02\x03", ("192.168.1.100", 60000))
    
    result = proposed_udp_ping("192.168.1.100", 60000)
    
    assert result is True
    mock_q.send.assert_called_once()
    mock_q.sock.settimeout.assert_called_with(1.0)
    mock_q.close.assert_called_once()


@patch("control.driver.quabo_driver.QUABO")
def test_proposed_udp_ping_timeout(mock_quabo_cls) -> None:
    """If the socket times out waiting for data, ping is False."""
    mock_q = MagicMock()
    mock_quabo_cls.return_value = mock_q
    
    # Simulate a timeout
    mock_q.sock.recvfrom.side_effect = TimeoutError("Socket timeout")
    
    result = proposed_udp_ping("192.168.1.100", 60000)
    
    assert result is False
    mock_q.send.assert_called_once()
    mock_q.close.assert_called_once()


@patch("control.driver.quabo_driver.QUABO")
def test_proposed_udp_ping_oserror(mock_quabo_cls) -> None:
    """If the socket throws an OSError (e.g. host unreachable), ping is False."""
    mock_q = MagicMock()
    mock_quabo_cls.return_value = mock_q
    
    # Simulate an OSError
    mock_q.sock.recvfrom.side_effect = OSError("No route to host")
    
    result = proposed_udp_ping("192.168.1.100", 60000)
    
    assert result is False
    mock_q.send.assert_called_once()
    mock_q.close.assert_called_once()
