"""
test_config_file_2.py

Unit tests for control/utils/config_file.py.
Covers IP math, string parsing, config loading from temp files, and
module→DAQ-node assignment utilities.  No hardware required.
"""

import json
from typing import Any

import pytest

from control.utils.config_file import (
    assign_numbers,
    get_modules,
    load_and_validate,
)
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    ObsConfig,
    QuaboUids,
)

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
