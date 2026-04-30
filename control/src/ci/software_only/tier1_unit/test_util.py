"""
test_util.py

Unit tests for control/utils/util.py.
Focuses on pure functions that need no hardware:
  - ip_addr_str_to_bytes
  - now_str
  - get_daemons / get_permanent_daemons (mocked config)
No hardware or network access required.
"""

import datetime

import pytest

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaemonConfig,
)
from control.utils.util import (
    get_daemons,
    get_permanent_daemons,
    ip_addr_str_to_bytes,
    now_str,
)

# ===========================================================================
# ip_addr_str_to_bytes
# ===========================================================================

class TestIpAddrStrToBytes:
    @pytest.mark.parametrize("ip, expected", [
        ("0.0.0.0",         bytearray([0, 0, 0, 0])),
        ("255.255.255.255", bytearray([255, 255, 255, 255])),
        ("192.168.1.100",   bytearray([192, 168, 1, 100])),
        ("10.0.0.1",        bytearray([10, 0, 0, 1])),
        ("172.16.254.1",    bytearray([172, 16, 254, 1])),
    ])
    def test_valid_ips(self, ip, expected) -> None:
        assert ip_addr_str_to_bytes(ip) == expected

    def test_returns_bytearray(self) -> None:
        result = ip_addr_str_to_bytes("192.168.1.100")
        assert isinstance(result, bytearray)

    def test_length_is_4(self) -> None:
        result = ip_addr_str_to_bytes("192.168.1.100")
        assert len(result) == 4

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = ip_addr_str_to_bytes("  192.168.1.100  ")
        assert result == bytearray([192, 168, 1, 100])

    def test_too_few_octets_raises(self) -> None:
        with pytest.raises(Exception, match="bad IP"):
            ip_addr_str_to_bytes("192.168.1")

    def test_too_many_octets_raises(self) -> None:
        with pytest.raises(Exception, match="bad IP"):
            ip_addr_str_to_bytes("192.168.1.1.1")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(Exception, match="bad IP addr"):
            ip_addr_str_to_bytes("")

    def test_octet_above_255_raises(self) -> None:
        with pytest.raises(Exception, match="bad IP"):
            ip_addr_str_to_bytes("192.168.1.256")

    def test_negative_octet_raises(self) -> None:
        with pytest.raises(Exception, match="bad IP"):
            ip_addr_str_to_bytes("192.168.1.-1")

    def test_non_numeric_octet_raises(self) -> None:
        with pytest.raises((Exception, ValueError)):
            ip_addr_str_to_bytes("192.168.abc.1")

    def test_first_octet_is_correct(self) -> None:
        result = ip_addr_str_to_bytes("10.20.30.40")
        assert result[0] == 10

    def test_last_octet_is_correct(self) -> None:
        result = ip_addr_str_to_bytes("10.20.30.40")
        assert result[3] == 40

    def test_round_trip_consistency(self) -> None:
        """Bytes can be reassembled into the original IP string."""
        ip = "192.168.3.200"
        b = ip_addr_str_to_bytes(ip)
        reconstructed = ".".join(str(x) for x in b)
        assert reconstructed == ip


# ===========================================================================
# now_str
# ===========================================================================

class TestNowStr:
    def test_returns_string(self) -> None:
        assert isinstance(now_str(), str)

    def test_non_empty(self) -> None:
        assert len(now_str()) > 0

    def test_parseable_as_datetime(self) -> None:
        s = now_str()
        # now_str() calls datetime.datetime.fromtimestamp(int(time.time())).isoformat()
        # Format: "2026-03-27T14:30:00" (no timezone, no subsecond)
        dt = datetime.datetime.fromisoformat(s)
        assert dt is not None

    def test_truncated_to_seconds(self) -> None:
        """now_str uses int(time.time()), so microseconds should always be 0."""
        s = now_str()
        dt = datetime.datetime.fromisoformat(s)
        assert dt.microsecond == 0

    def test_is_recent(self) -> None:
        """now_str should be within 5 seconds of the actual current time."""
        s = now_str()
        dt = datetime.datetime.fromisoformat(s)
        now = datetime.datetime.now().replace(microsecond=0)
        diff = abs((now - dt).total_seconds())
        assert diff < 5


# ===========================================================================
# get_daemons  (mocked config_file dependency)
# ===========================================================================

class TestGetDaemons:
    def test_always_includes_store_influxdb(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config", lambda: None)
        result = get_daemons()
        assert str(PanoPaths.daemons_dir() / "storeInfluxDB.py") in result

    def test_enabled_daemon_is_included(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={"hk": True},
                                permanent_daemons={}
                            ))
        result = get_daemons()
        assert str(PanoPaths.daemons_dir() / "capture_hk.py") in result

    def test_disabled_daemon_is_excluded(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={"hk": True, "gps": False},
                                permanent_daemons={}
                            ))
        result = get_daemons()
        assert str(PanoPaths.daemons_dir() / "capture_gps.py") not in result

    def test_multiple_enabled_daemons(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={"hk": True, "gps": True, "wr": True},
                                permanent_daemons={}
                            ))
        result = get_daemons()
        assert str(PanoPaths.daemons_dir() / "capture_hk.py") in result
        assert str(PanoPaths.daemons_dir() / "capture_gps.py") in result
        assert str(PanoPaths.daemons_dir() / "capture_wr.py") in result

    def test_no_daemons_key_returns_only_base(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={},
                                permanent_daemons={"mount": True}
                            ))
        result = get_daemons()
        # 'daemons' empty → only base list
        assert result == [str(PanoPaths.daemons_dir() / "storeInfluxDB.py")]

    def test_does_not_mutate_base_list(self, monkeypatch) -> None:
        """Calling get_daemons twice returns consistent results (no global mutation)."""
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={"hk": True},
                                permanent_daemons={}
                            ))
        r1 = get_daemons()
        r2 = get_daemons()
        assert r1 == r2

    def test_returns_list(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config", lambda: None)
        assert isinstance(get_daemons(), list)

    def test_all_entries_are_strings(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={"hk": True, "gps": True},
                                permanent_daemons={}
                            ))
        for entry in get_daemons():
            assert isinstance(entry, str)


# ===========================================================================
# get_permanent_daemons
# ===========================================================================

class TestGetPermanentDaemons:
    def test_always_includes_store_influxdb(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config", lambda: None)
        result = get_permanent_daemons()
        assert str(PanoPaths.daemons_dir() / "storeInfluxDB.py") in result

    def test_enabled_permanent_daemon_is_included(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={},
                                permanent_daemons={"mount": True}
                            ))
        result = get_permanent_daemons()
        assert str(PanoPaths.daemons_dir() / "permanent_mount.py") in result

    def test_disabled_permanent_daemon_excluded(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={},
                                permanent_daemons={"mount": False}
                            ))
        result = get_permanent_daemons()
        assert str(PanoPaths.daemons_dir() / "permanent_mount.py") not in result

    def test_empty_config_returns_only_base(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config", lambda: None)
        result = get_permanent_daemons()
        assert result == [
            str(PanoPaths.daemons_dir() / "storeInfluxDB.py"),
            str(PanoPaths.daemons_dir() / "storeLoki.py"),
        ]

    def test_multiple_permanent_daemons(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={},
                                permanent_daemons={
                                    "mount": True, "dome": True, "alerts": False
                                }
                            ))
        result = get_permanent_daemons()
        assert str(PanoPaths.daemons_dir() / "permanent_mount.py") in result
        assert str(PanoPaths.daemons_dir() / "permanent_dome.py") in result
        assert str(PanoPaths.daemons_dir() / "permanent_alerts.py") not in result

    def test_does_not_mutate_on_repeated_calls(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config",
                            lambda: DaemonConfig(
                                daemons={},
                                permanent_daemons={"mount": True}
                            ))
        r1 = get_permanent_daemons()
        r2 = get_permanent_daemons()
        assert r1 == r2

    def test_returns_list(self, monkeypatch) -> None:
        monkeypatch.setattr("control.utils.util._safe_get_daemons_config", lambda: None)
        assert isinstance(get_permanent_daemons(), list)

