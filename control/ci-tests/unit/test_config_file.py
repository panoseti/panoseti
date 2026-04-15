"""
test_config_file.py

Unit tests for control/utils/config_file.py.
Covers IP math, string parsing, config loading from temp files, and
module→DAQ-node assignment utilities.  No hardware required.
"""

import json

import pytest

from utils.config_file import (
    assign_numbers,
    expand_ranges,
    get_boardloc,
    get_modules,
    ip_addr_to_module_id,
    load_and_validate,
    module_id_to_daq_node,
    quabo_ip_addr,
    string_to_list,
)
from utils.pydantic_config_models import DataConfigValidator

# ===========================================================================
# ip_addr_to_module_id
# Formula: module_id = (ip_octet3 * 256 + ip_octet4) >> 2 & 0xFF
# ===========================================================================

class TestIpAddrToModuleId:
    @pytest.mark.parametrize("ip, expected", [
        ("192.168.0.0",   0),    # n=0, 0>>2=0
        ("192.168.0.4",   1),    # n=4, 4>>2=1
        ("192.168.0.8",   2),    # n=8, 8>>2=2
        ("192.168.0.252", 63),   # n=252, 252>>2=63
        ("192.168.1.0",   64),   # n=256, 256>>2=64
        ("192.168.3.200", 242),  # n=968, 968>>2=242
        ("192.168.3.252", 255),  # n=1020, 1020>>2=255 — max valid
    ])
    def test_known_ips(self, ip, expected):
        assert ip_addr_to_module_id(ip) == expected

    def test_module_id_is_in_0_255_range(self):
        """All valid quabo base IPs must map to 0-255."""
        for octet3 in range(4):
            for octet4_base in range(0, 256, 4):  # only multiples of 4 are valid module base IPs
                ip = f"192.168.{octet3}.{octet4_base}"
                mid = ip_addr_to_module_id(ip)
                assert 0 <= mid <= 255, f"{ip} → {mid} out of range"

    def test_boardloc_zero_quabo_equals_module_id_times_4(self):
        """BOARDLOC for quabo 0 must equal module_id * 4."""
        ip = "192.168.3.200"
        mid = ip_addr_to_module_id(ip)
        bl = get_boardloc(ip, 0)
        assert bl == mid * 4


# ===========================================================================
# quabo_ip_addr
# ===========================================================================

class TestQuaboIpAddr:
    def test_quabo_0_is_base_ip(self):
        assert quabo_ip_addr("192.168.3.200", 0) == "192.168.3.200"

    def test_quabo_1(self):
        assert quabo_ip_addr("192.168.3.200", 1) == "192.168.3.201"

    def test_quabo_2(self):
        assert quabo_ip_addr("192.168.3.200", 2) == "192.168.3.202"

    def test_quabo_3(self):
        assert quabo_ip_addr("192.168.3.200", 3) == "192.168.3.203"

    def test_different_subnet(self):
        assert quabo_ip_addr("10.0.1.4", 2) == "10.0.1.6"

    def test_returns_string(self):
        result = quabo_ip_addr("192.168.3.200", 1)
        assert isinstance(result, str)


# ===========================================================================
# get_boardloc
# Formula: boardloc = ip_octet3 * 256 + ip_octet4 + quabo_index
# ===========================================================================

class TestGetBoardloc:
    def test_quabo_0(self):
        # 3 * 256 + 200 + 0 = 968
        assert get_boardloc("192.168.3.200", 0) == 968

    def test_quabo_1(self):
        assert get_boardloc("192.168.3.200", 1) == 969

    def test_quabo_3(self):
        assert get_boardloc("192.168.3.200", 3) == 971

    def test_different_ip(self):
        # 0 * 256 + 4 + 0 = 4
        assert get_boardloc("192.168.0.4", 0) == 4

    def test_boardloc_equals_module_id_times_4_plus_quabo_index(self):
        ip = "192.168.3.200"
        mid = ip_addr_to_module_id(ip)
        for q in range(4):
            assert get_boardloc(ip, q) == mid * 4 + q


# ===========================================================================
# string_to_list
# Parses comma-separated ranges like "0-2, 5-6" into [0,1,2,5,6]
# ===========================================================================

class TestStringToList:
    @pytest.mark.parametrize("s, expected", [
        ("0",       [0]),
        ("5",       [5]),
        ("0-3",     [0, 1, 2, 3]),
        ("128-130", [128, 129, 130]),
        ("0,5,10",  [0, 5, 10]),
        ("0-2, 5-6", [0, 1, 2, 5, 6]),   # space after comma is handled by int()
        ("224-231", list(range(224, 232))),
        ("255",     [255]),
    ])
    def test_various_formats(self, s, expected):
        assert string_to_list(s) == expected

    def test_single_element_range(self):
        """A range "N-N" yields a single-element list."""
        assert string_to_list("5-5") == [5]

    def test_preserves_order(self):
        """Output order matches the specification order."""
        result = string_to_list("10,5,0")
        assert result == [10, 5, 0]


# ===========================================================================
# expand_ranges
# Mutates daq_config['daq_nodes'][*]['module_ids'] in-place
# ===========================================================================

class TestExpandRanges:
    def test_string_range_becomes_list(self):
        config = {"daq_nodes": [{"module_ids": "224-225"}]}
        expand_ranges(config)
        assert config["daq_nodes"][0]["module_ids"] == [224, 225]

    def test_single_value_string(self):
        config = {"daq_nodes": [{"module_ids": "0"}]}
        expand_ranges(config)
        assert config["daq_nodes"][0]["module_ids"] == [0]

    def test_already_list_is_preserved(self):
        config = {"daq_nodes": [{"module_ids": [0, 1, 2]}]}
        expand_ranges(config)
        assert set(config["daq_nodes"][0]["module_ids"]) == {0, 1, 2}

    def test_multiple_nodes(self):
        config = {
            "daq_nodes": [
                {"module_ids": "0-1"},
                {"module_ids": "128-129"},
            ]
        }
        expand_ranges(config)
        assert config["daq_nodes"][0]["module_ids"] == [0, 1]
        assert config["daq_nodes"][1]["module_ids"] == [128, 129]

    def test_invalid_type_raises(self):
        config = {"daq_nodes": [{"module_ids": 42}]}
        with pytest.raises((ValueError, TypeError)):
            expand_ranges(config)


# ===========================================================================
# module_id_to_daq_node
# ===========================================================================

class TestModuleIdToDaqNode:
    @pytest.fixture
    def expanded_config(self):
        config = {
            "daq_nodes": [
                {"ip_addr": "10.0.0.2", "module_ids": "224-225"},
                {"ip_addr": "10.0.0.3", "module_ids": "226-227"},
            ]
        }
        expand_ranges(config)
        return config

    def test_finds_correct_node_for_first_range(self, expanded_config):
        node = module_id_to_daq_node(expanded_config, 224)
        assert node["ip_addr"] == "10.0.0.2"

    def test_finds_correct_node_for_second_range(self, expanded_config):
        node = module_id_to_daq_node(expanded_config, 226)
        assert node["ip_addr"] == "10.0.0.3"

    def test_missing_module_id_raises(self, expanded_config):
        with pytest.raises(Exception, match="no DAQ node"):
            module_id_to_daq_node(expanded_config, 999)

    def test_boundary_values(self, expanded_config):
        assert module_id_to_daq_node(expanded_config, 225)["ip_addr"] == "10.0.0.2"
        assert module_id_to_daq_node(expanded_config, 227)["ip_addr"] == "10.0.0.3"


# ===========================================================================
# assign_numbers
# Injects 'num' into domes and 'id' into modules (= module_id from IP)
# ===========================================================================

class TestAssignNumbers:
    def test_dome_num_starts_at_zero(self):
        config = {
            "domes": [
                {"modules": [{"ip_addr": "192.168.3.200"}]},
                {"modules": [{"ip_addr": "192.168.3.204"}]},
            ]
        }
        assign_numbers(config)
        assert config["domes"][0]["num"] == 0
        assert config["domes"][1]["num"] == 1

    def test_module_id_is_correct(self):
        config = {"domes": [{"modules": [{"ip_addr": "192.168.3.200"}]}]}
        assign_numbers(config)
        assert config["domes"][0]["modules"][0]["id"] == 242

    def test_multiple_modules_in_one_dome(self):
        config = {
            "domes": [
                {
                    "modules": [
                        {"ip_addr": "192.168.3.200"},
                        {"ip_addr": "192.168.3.204"},
                    ]
                }
            ]
        }
        assign_numbers(config)
        assert config["domes"][0]["modules"][0]["id"] == 242
        assert config["domes"][0]["modules"][1]["id"] == 243

    def test_does_not_change_other_fields(self):
        config = {"domes": [{"name": "dome0", "modules": [{"ip_addr": "192.168.3.200", "extra": "x"}]}]}
        assign_numbers(config)
        assert config["domes"][0]["name"] == "dome0"
        assert config["domes"][0]["modules"][0]["extra"] == "x"


# ===========================================================================
# get_modules
# ===========================================================================

class TestGetModules:
    def test_returns_flat_list_of_modules(self):
        obs_config = {
            "domes": [
                {"modules": [{"ip_addr": "192.168.3.200"}, {"ip_addr": "192.168.3.204"}]},
                {"modules": [{"ip_addr": "192.168.3.208"}]},
            ]
        }
        modules = get_modules(obs_config)
        assert len(modules) == 3
        ips = [m["ip_addr"] for m in modules]
        assert "192.168.3.200" in ips
        assert "192.168.3.208" in ips

    def test_single_dome_single_module(self):
        obs_config = {"domes": [{"modules": [{"ip_addr": "192.168.3.200"}]}]}
        assert len(get_modules(obs_config)) == 1

    def test_empty_domes(self):
        assert get_modules({"domes": []}) == []


# ===========================================================================
# load_and_validate (integration with temp files)
# ===========================================================================

class TestLoadAndValidate:
    def _write_json(self, tmp_path, relpath, data):
        """Write JSON to a subdirectory of tmp_path matching the expected config path."""
        full = tmp_path / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(data))
        return str(tmp_path)

    def test_loads_valid_data_config(self, tmp_path, minimal_data_config):
        base = self._write_json(tmp_path, "configs/data_config.json", minimal_data_config)
        result = load_and_validate(DataConfigValidator, "configs/data_config.json", base, "Data Config")
        assert result["run_type"] == "sci"

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Missing file"):
            load_and_validate(DataConfigValidator, "configs/data_config.json",
                              str(tmp_path), "Data Config")

    def test_invalid_json_raises_value_error(self, tmp_path):
        p = tmp_path / "configs"
        p.mkdir()
        (p / "data_config.json").write_text("{ not valid json }")
        with pytest.raises(ValueError, match="JSON Parse Error"):
            load_and_validate(DataConfigValidator, "configs/data_config.json",
                              str(tmp_path), "Data Config")

    def test_schema_error_exits_cleanly(self, tmp_path):
        """Invalid schema causes sys.exit in non-CLI mode (by default)."""
        bad_data = {"run_type": "a" * 20}  # Too long — Pydantic will reject
        self._write_json(tmp_path, "configs/data_config.json", bad_data)
        # Default behaviour is sys.exit(1) on schema error when not in CLI mode
        with pytest.raises(SystemExit):
            load_and_validate(DataConfigValidator, "configs/data_config.json",
                              str(tmp_path), "Data Config")
