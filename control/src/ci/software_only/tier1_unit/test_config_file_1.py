"""
test_config_file_1.py

Unit tests for control/utils/config_file.py.
Covers IP math, string parsing, config loading from temp files, and
module→DAQ-node assignment utilities.  No hardware required.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from control.utils.config_file import (
    expand_ranges,
    get_boardloc,
    ip_addr_to_module_id,
    module_id_to_daq_node,
    quabo_ip_addr,
    string_to_list,
)
from control.utils.pydantic_config_models import (
    DaqConfig,
)

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
    def test_known_ips(self, ip, expected) -> None:
        assert ip_addr_to_module_id(ip) == expected

    def test_module_id_is_in_0_255_range(self) -> None:
        """All valid quabo base IPs must map to 0-255."""
        for octet3 in range(4):
            for octet4_base in range(0, 256, 4):  # only multiples of 4 are valid module base IPs
                ip = f"192.168.{octet3}.{octet4_base}"
                mid = ip_addr_to_module_id(ip)
                assert 0 <= mid <= 255, f"{ip} → {mid} out of range"

    def test_boardloc_zero_quabo_equals_module_id_times_4(self) -> None:
        """BOARDLOC for quabo 0 must equal module_id * 4."""
        ip = "192.168.3.200"
        mid = ip_addr_to_module_id(ip)
        bl = get_boardloc(ip, 0)
        assert bl == mid * 4


# ===========================================================================
# quabo_ip_addr
# ===========================================================================

class TestQuaboIpAddr:
    def test_quabo_0_is_base_ip(self) -> None:
        assert quabo_ip_addr("192.168.3.200", 0) == "192.168.3.200"

    def test_quabo_1(self) -> None:
        assert quabo_ip_addr("192.168.3.200", 1) == "192.168.3.201"

    def test_quabo_2(self) -> None:
        assert quabo_ip_addr("192.168.3.200", 2) == "192.168.3.202"

    def test_quabo_3(self) -> None:
        assert quabo_ip_addr("192.168.3.200", 3) == "192.168.3.203"

    def test_different_subnet(self) -> None:
        assert quabo_ip_addr("10.0.1.4", 2) == "10.0.1.6"

    def test_returns_string(self) -> None:
        result = quabo_ip_addr("192.168.3.200", 1)
        assert isinstance(result, str)


# ===========================================================================
# get_boardloc
# Formula: boardloc = ip_octet3 * 256 + ip_octet4 + quabo_index
# ===========================================================================

class TestGetBoardloc:
    def test_quabo_0(self) -> None:
        # 3 * 256 + 200 + 0 = 968
        assert get_boardloc("192.168.3.200", 0) == 968

    def test_quabo_1(self) -> None:
        assert get_boardloc("192.168.3.200", 1) == 969

    def test_quabo_3(self) -> None:
        assert get_boardloc("192.168.3.200", 3) == 971

    def test_different_ip(self) -> None:
        # 0 * 256 + 4 + 0 = 4
        assert get_boardloc("192.168.0.4", 0) == 4

    def test_boardloc_equals_module_id_times_4_plus_quabo_index(self) -> None:
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
    def test_various_formats(self, s, expected) -> None:
        assert string_to_list(s) == expected

    def test_single_element_range(self) -> None:
        """A range "N-N" yields a single-element list."""
        assert string_to_list("5-5") == [5]

    def test_preserves_order(self) -> None:
        """Output order matches the specification order."""
        result = string_to_list("10,5,0")
        assert result == [10, 5, 0]


# ===========================================================================
# expand_ranges
# Mutates daq_config['daq_nodes'][*]['module_ids'] in-place
# ===========================================================================

class TestExpandRanges:
    def make_valid_daq_model(self, nodes: list[dict[str, Any]]) -> DaqConfig:
        """Helper to create a DaqConfig model with all required fields."""
        valid_nodes = []
        for i, node in enumerate(nodes):
            full_node = {
                "username": "root",
                "data_dir": "/data",
                "ip_addr": f"192.168.0.{10+i}",
                **node
            }
            valid_nodes.append(full_node)
            
        config_dict = {
            "head_node_data_dir": "/data/head",
            "head_node_ip_addr": "10.0.1.5",
            "daq_nodes": valid_nodes
        }
        return DaqConfig(**config_dict)

    def test_string_range_becomes_list(self) -> None:
        model = self.make_valid_daq_model([{"module_ids": "224-225"}])
        expand_ranges(model)
        assert model.daq_nodes[0].module_ids == [224, 225]

    def test_single_value_string(self) -> None:
        model = self.make_valid_daq_model([{"module_ids": "0"}])
        expand_ranges(model)
        assert model.daq_nodes[0].module_ids == [0]

    def test_already_list_is_preserved(self) -> None:
        model = self.make_valid_daq_model([{"module_ids": [0, 1, 2]}])
        expand_ranges(model)
        assert set(model.daq_nodes[0].module_ids) == {0, 1, 2}

    def test_multiple_nodes(self) -> None:
        model = self.make_valid_daq_model([
            {"module_ids": "0-1"},
            {"module_ids": "128-129"},
        ])
        expand_ranges(model)
        assert model.daq_nodes[0].module_ids == [0, 1]
        assert model.daq_nodes[1].module_ids == [128, 129]

    def test_invalid_type_raises(self) -> None:
        # Pydantic validation will catch this during instantiation or if manually assigned
        with pytest.raises(ValidationError):
            self.make_valid_daq_model([{"module_ids": 42.5}])


# ===========================================================================
# module_id_to_daq_node
# ===========================================================================

class TestModuleIdToDaqNode:
    @pytest.fixture
    def expanded_config(self) -> DaqConfig:
        config_dict: dict[str, Any] = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {
                    "username": "panoseti",
                    "data_dir": "/data",
                    "ip_addr": "10.0.0.2",
                    "module_ids": "224-225"
                },
                {
                    "username": "panoseti",
                    "data_dir": "/data",
                    "ip_addr": "10.0.0.3",
                    "module_ids": "226-227"
                },
            ]
        }
        # Model handles expansion automatically via validators
        return DaqConfig(**config_dict)

    def test_finds_correct_node_for_first_range(self, expanded_config) -> None:
        node = module_id_to_daq_node(expanded_config, 224)
        assert str(node.ip_addr) == "10.0.0.2"

    def test_finds_correct_node_for_second_range(self, expanded_config) -> None:
        node = module_id_to_daq_node(expanded_config, 226)
        assert str(node.ip_addr) == "10.0.0.3"

    def test_missing_module_id_raises(self, expanded_config) -> None:
        with pytest.raises(Exception, match="no DAQ node"):
            module_id_to_daq_node(expanded_config, 999)

    def test_boundary_values(self, expanded_config) -> None:
        assert str(module_id_to_daq_node(expanded_config, 225).ip_addr) == "10.0.0.2"
        assert str(module_id_to_daq_node(expanded_config, 227).ip_addr) == "10.0.0.3"
