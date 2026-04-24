"""
test_config_file.py

Unit tests for control/utils/config_file.py.
Covers IP math, string parsing, config loading from temp files, and
module→DAQ-node assignment utilities.  No hardware required.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from control.utils.config_file import (
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
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    ObsConfig,
    QuaboUids,
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


# ===========================================================================
# assign_numbers
# Injects 'num' into domes and 'id' into modules (= module_id from IP)
# ===========================================================================

class TestAssignNumbers:
    def make_valid_obs_config(self, domes_data: list[dict[str, Any]]) -> ObsConfig:
        domes = []
        for i, d in enumerate(domes_data):
            modules = []
            for m in d.get("modules", []):
                modules.append({
                    "ip_addr": m.get("ip_addr", "192.168.0.0"),
                    "mobo_serialno": "S123",
                    "quabo_version": "qfp",
                    **m
                })
            
            # Create dome dict, ensuring our processed modules list is used
            dome_dict = {
                "name": f"dome{i}",
                "obslat": 34.0,
                "obslon": -116.0,
                "obsalt": 1700.0,
                **d
            }
            dome_dict["modules"] = modules
            domes.append(dome_dict)
        return ObsConfig(name="Palomar", domes=domes)

    def test_dome_num_starts_at_zero(self) -> None:
        model = self.make_valid_obs_config([
            {"modules": [{"ip_addr": "192.168.3.200"}]},
            {"modules": [{"ip_addr": "192.168.3.204"}]},
        ])
        assign_numbers(model)
        assert model.domes[0].num == 0
        assert model.domes[1].num == 1

    def test_module_id_is_correct(self) -> None:
        model = self.make_valid_obs_config([{"modules": [{"ip_addr": "192.168.3.200"}]}])
        assign_numbers(model)
        assert model.domes[0].modules[0].id == 242

    def test_multiple_modules_in_one_dome(self) -> None:
        model = self.make_valid_obs_config([
            {
                "modules": [
                    {"ip_addr": "192.168.3.200"},
                    {"ip_addr": "192.168.3.204"},
                ]
            }
        ])
        assign_numbers(model)
        assert model.domes[0].modules[0].id == 242
        assert model.domes[0].modules[1].id == 243

    def test_does_not_change_other_fields(self) -> None:
        model = self.make_valid_obs_config([{"name": "dome0", "modules": [{"ip_addr": "192.168.3.200", "position_angle": 45.0}]}])
        assign_numbers(model)
        assert model.domes[0].name == "dome0"
        assert model.domes[0].modules[0].position_angle == 45.0

    def test_assign_numbers_no_domes(self) -> None:
        model = ObsConfig(name="test", domes=[])
        assign_numbers(model)
        assert len(model.domes) == 0


# ===========================================================================
# get_modules
# ===========================================================================

class TestGetModules:
    def test_returns_flat_list_of_modules(self) -> None:
        obs_config_dict = {
            "name": "test",
            "domes": [
                {
                    "name": "d1", "obslat": 0, "obslon": 0, "obsalt": 0,
                    "modules": [
                        {"ip_addr": "192.168.3.200", "mobo_serialno": "s1", "quabo_version": "q"},
                        {"ip_addr": "192.168.3.204", "mobo_serialno": "s2", "quabo_version": "q"}
                    ]
                },
                {
                    "name": "d2", "obslat": 0, "obslon": 0, "obsalt": 0,
                    "modules": [{"ip_addr": "192.168.3.208", "mobo_serialno": "s3", "quabo_version": "q"}]
                },
            ]
        }
        from control.utils.pydantic_config_models import ObsConfig
        obs_config = ObsConfig(**obs_config_dict)
        modules = get_modules(obs_config)
        assert len(modules) == 3
        ips = [str(m.ip_addr) for m in modules]
        assert "192.168.3.200" in ips
        assert "192.168.3.208" in ips

    def test_single_dome_single_module(self) -> None:
        obs_config_dict = {
            "name": "test",
            "domes": [
                {
                    "name": "d1", "obslat": 0, "obslon": 0, "obsalt": 0,
                    "modules": [{"ip_addr": "192.168.3.200", "mobo_serialno": "s1", "quabo_version": "q"}]
                }
            ]
        }
        from control.utils.pydantic_config_models import ObsConfig
        obs_config = ObsConfig(**obs_config_dict)
        assert len(get_modules(obs_config)) == 1

    def test_empty_domes(self) -> None:
        from control.utils.pydantic_config_models import ObsConfig
        assert get_modules(ObsConfig(name="test", domes=[])) == []



# ===========================================================================
# load_and_validate (integration with temp files)
# ===========================================================================

class TestLoadAndValidate:
    def _write_json(self, tmp_path, relpath, data):
        """Write JSON to a subdirectory of tmp_path matching the expected config path."""
        full = tmp_path / "unit_test_configs" / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(data))
        return str(tmp_path / "unit_test_configs")

    def test_loads_valid_data_config(self, tmp_path, minimal_data_config) -> None:
        base = self._write_json(tmp_path, "configs/data_config.json", minimal_data_config)
        result = load_and_validate(DataConfig, "configs/data_config.json", base, "Data Config")
        assert result.run_type == "sci"

    def test_missing_file_raises_value_error(self, tmp_path) -> None:
        # Use a path that is definitely empty
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="Missing file"):
            load_and_validate(DataConfig, "configs/data_config.json",
                              str(empty_dir), "Data Config")

    def test_invalid_json_raises_value_error(self, tmp_path) -> None:
        p = tmp_path / "unit_test_configs" / "configs"
        p.mkdir(parents=True, exist_ok=True)
        (p / "data_config.json").write_text("{ not valid json }")
        with pytest.raises(ValueError, match="JSON Parse Error"):
            load_and_validate(DataConfig, "configs/data_config.json",
                              str(tmp_path / "unit_test_configs"), "Data Config")

    def test_schema_error_raises_value_error(self, tmp_path) -> None:
        """Invalid schema causes ValueError in non-CLI mode (by default)."""
        bad_data = {"run_type": "a" * 20}  # Too long — Pydantic will reject
        base = self._write_json(tmp_path, "configs/data_config.json", bad_data)
        # Default behaviour is now ValueError on schema error when not in CLI mode
        # to allow orchestration rollback ladders to run.
        with pytest.raises(ValueError, match="Pydantic Validation failed"):
            load_and_validate(DataConfig, "configs/data_config.json",
                              base, "Data Config")


# ===========================================================================
# associate
# ===========================================================================

class TestAssociate:
    def test_associate_links_modules_and_nodes(self) -> None:
        daq_config_dict = {
            "head_node_data_dir": "/data/head",
            "head_node_ip_addr": "10.0.1.5",
            "daq_nodes": [
                {
                    "username": "root",
                    "data_dir": "/data",
                    "ip_addr": "10.0.1.10",
                    "module_ids": [1, 2]
                }
            ]
        }
        daq_config = DaqConfig(**daq_config_dict)
        
        quabo_uids_dict = {
            "domes": [
                {
                    "modules": [
                        {
                            "ip_addr": "192.168.0.4", # module_id = 1
                            "quabos": [{"uid": "a"}, {"uid": "b"}, {"uid": "c"}, {"uid": "d"}]
                        }
                    ]
                }
            ]
        }
        quabo_uids = QuaboUids(**quabo_uids_dict)

        # assign_numbers injects 'id' which is needed by associate -> module_id_to_daq_node
        from control.utils.config_file import assign_numbers, associate
        assign_numbers(quabo_uids)
        
        associate(daq_config, quabo_uids)
        
        # Check node -> modules link
        assert len(daq_config.daq_nodes[0].modules) == 1
        assert str(daq_config.daq_nodes[0].modules[0].ip_addr) == "192.168.0.4"
        
        # Check module -> node link
        assert quabo_uids.domes[0].modules[0].daq_node == daq_config.daq_nodes[0]

# ===========================================================================
# get_module_quabo_uids
# ===========================================================================

class TestGetModuleQuaboUids:
    def test_get_mapping(self) -> None:
        from control.utils.config_file import get_module_quabo_uids
        uids_dict = {
            "domes": [
                {
                    "modules": [
                        {
                            "ip_addr": "192.168.0.4",
                            "quabos": [{"uid": "u1"}, {"uid": "u2"}, {"uid": "u3"}, {"uid": "u4"}]
                        }
                    ]
                }
            ]
        }
        quabo_uids = QuaboUids(**uids_dict)
        res = get_module_quabo_uids(quabo_uids)
        assert res == {"192.168.0.4": ["u1", "u2", "u3", "u4"]}

# ===========================================================================
# check_config_file
# ===========================================================================

class TestCheckConfigFile:
    def test_missing_file_exits(self, tmp_path) -> None:
        from control.utils.config_file import check_config_file
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            check_config_file("nonexistent.json", tmp_path)
        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == 1

    def test_existing_file_passes(self, tmp_path) -> None:
        from control.utils.config_file import check_config_file
        p = tmp_path / "exists.json"
        p.write_text("{}")
        # Should not raise
        check_config_file("exists.json", tmp_path)

# ===========================================================================
# Configuration Loaders (get_obs_config, get_daq_config, etc.)
# ===========================================================================

class TestGetConfigs:
    def _write_json(self, tmp_path, filename, data):
        p = tmp_path / filename
        p.write_text(json.dumps(data))
        return str(tmp_path)

    def test_get_obs_config(self, tmp_path, minimal_obs_config) -> None:
        from control.utils.config_file import get_obs_config
        base = self._write_json(tmp_path, "obs_config.json", minimal_obs_config)
        config = get_obs_config(base)
        assert config.name == minimal_obs_config["name"]

    def test_get_daq_config(self, tmp_path, minimal_daq_config) -> None:
        from control.utils.config_file import get_daq_config
        base = self._write_json(tmp_path, "daq_config.json", minimal_daq_config)
        config = get_daq_config(base)
        assert str(config.head_node_ip_addr) == minimal_daq_config["head_node_ip_addr"]

    def test_get_data_config(self, tmp_path, minimal_data_config) -> None:
        from control.utils.config_file import get_data_config
        base = self._write_json(tmp_path, "data_config.json", minimal_data_config)
        config = get_data_config(base)
        assert config.run_type == minimal_data_config["run_type"]

    def test_get_network_config(self, tmp_path) -> None:
        from control.utils.config_file import get_network_config
        net_data: dict[str, Any] = {"modules": [], "daq_nodes": []}
        base = self._write_json(tmp_path, "network_config.json", net_data)
        config = get_network_config(base)
        assert len(config.modules) == 0

    def test_get_firmware_config(self, tmp_path, minimal_firmware_config) -> None:
        from control.utils.config_file import get_firmware_config
        base = self._write_json(tmp_path, "firmware.json", minimal_firmware_config)
        config = get_firmware_config(base)
        assert config.qfp == minimal_firmware_config["qfp"]

    def test_get_daemons_config(self, tmp_path) -> None:
        from control.utils.config_file import get_daemons_config
        daemons_data = {
            "daemons": {"hk": True},
            "permanent_daemons": {"influx": True}
        }
        base = self._write_json(tmp_path, "daemons.json", daemons_data)
        config = get_daemons_config(base)
        assert config.daemons.model_extra["hk"] is True

    def test_get_quabo_uids(self, tmp_path, monkeypatch) -> None:
        from control.utils.config_file import get_quabo_uids, quabo_uids_filename
        from control.utils.paths import PanoPaths
        
        uids_data = {
            "domes": [
                {
                    "modules": [
                        {
                            "ip_addr": "192.168.0.4",
                            "quabos": [{"uid": "u1"}, {"uid": "u2"}, {"uid": "u3"}, {"uid": "u4"}]
                        }
                    ]
                }
            ]
        }
        
        # Mock PanoPaths.tmp_dir to use our tmp_path
        monkeypatch.setattr(PanoPaths, "tmp_dir", lambda: tmp_path)
        p = tmp_path / quabo_uids_filename
        p.write_text(json.dumps(uids_data))
        
        config = get_quabo_uids()
        assert str(config.domes[0].modules[0].ip_addr) == "192.168.0.4"

    def test_get_quabo_info(self, tmp_path, monkeypatch) -> None:
        from control.utils.config_file import get_quabo_info, quabo_info_filename
        from control.utils.paths import PanoPaths
        
        info_data = [
            {"uid": "u1", "rev": "1"},
            {"uid": "u2", "rev": "2"}
        ]
        
        monkeypatch.setattr(PanoPaths, "quabos_dir", lambda: tmp_path)
        p = tmp_path / quabo_info_filename
        p.write_text(json.dumps(info_data))
        
        info = get_quabo_info()
        assert info["u1"]["rev"] == "1"
        assert info["u2"]["rev"] == "2"

    def test_get_quabo_ph_baselines(self, tmp_path, monkeypatch) -> None:
        from control.utils.config_file import get_quabo_ph_baselines, quabo_ph_baseline_filename
        from control.utils.paths import PanoPaths
        
        baseline_data = {"u1": [100, 101]}
        
        monkeypatch.setattr(PanoPaths, "tmp_dir", lambda: tmp_path)
        p = tmp_path / quabo_ph_baseline_filename
        p.write_text(json.dumps(baseline_data))
        
        baselines = get_quabo_ph_baselines()
        assert baselines["u1"] == [100, 101]

    def test_get_detector_info(self, tmp_path, monkeypatch) -> None:
        from control.utils.config_file import (
            data_config_filename,
            detector_info_filename,
            get_detector_info,
            obs_config_filename,
        )
        from control.utils.paths import PanoPaths
        
        det_data = [{"serialno": "d1", "operating_voltage": 75.0}]
        obs_data = {"name": "test", "domes": []}
        data_data = {"run_type": "sci", "detector_overvoltage": 3}
        
        monkeypatch.setattr(PanoPaths, "quabos_dir", lambda: tmp_path)
        monkeypatch.setattr(PanoPaths, "config_dir", lambda: tmp_path)
        
        (tmp_path / detector_info_filename).write_text(json.dumps(det_data))
        (tmp_path / obs_config_filename).write_text(json.dumps(obs_data))
        (tmp_path / data_config_filename).write_text(json.dumps(data_data))
        
        info = get_detector_info()
        assert info["d1"] == 75.0
