"""
ci/tier2_logic/test_config_logic.py

Logic tests for global configuration consistency.
Verifies cross-file invariants (BOARDLOC uniqueness, IP collisions, module-to-node mapping).
"""

from __future__ import annotations

import pytest

from control.topology.fleet import generate_palomar_topology
from control.utils.global_validator import GlobalConfigValidator


@pytest.fixture
def palomar_setup():
    return generate_palomar_topology()

def test_when_palomar_validated_then_global_invariants_pass(palomar_setup):
    """
    Intent: Verify that the realistic Palomar topology satisfies all global invariants.
    """
    daq, uids, net, obs = palomar_setup
    configs = {
        'obs': obs,
        'data': None, # Not strictly needed for topology invariants
        'daq': daq,
        'network': net,
        'firmware': None,
        'uids': uids
    }
    validator = GlobalConfigValidator(configs)
    passed = validator.validate_all_rules()
    assert passed is True

def test_when_boardloc_collides_across_domes_then_rejected(palomar_setup):
    """
    Intent: Ensure BOARDLOC uniqueness is enforced even across different domes.
    Scenario: Add a module in a new dome with an IP that produces a duplicate module_id.
    Assertion: GlobalConfigValidator.validate_all_rules() returns False.
    """
    daq, uids, net, obs = palomar_setup
    
    # Gattini site has module_id 1 (192.168.3.248)
    # Add a module in PTI dome that also has module_id 1
    new_mod = obs.domes[3].modules[0].model_copy(update={"ip_addr": "192.168.3.248", "id": 1})
    obs.domes[3].modules.append(new_mod)
    
    # We must also update uids to match the new obs module
    uids.domes[0].modules.append(uids.domes[0].modules[0].model_copy(update={"id": 1}))

    configs = {'obs': obs, 'daq': daq, 'network': net, 'uids': uids}
    validator = GlobalConfigValidator(configs)
    passed = validator.validate_all_rules()
    
    assert passed is False
    assert any("collision" in t["info"].lower() or "module_id" in t["info"].lower() 
               for t in validator.report.tests if t["status"] == "ERROR")

def test_when_module_ids_overlap_across_daqnodes_then_rejected(palomar_setup):
    """
    Intent: Verify that two DAQ nodes cannot claim the same module_id.
    Scenario: Node B is modified to claim a module_id already handled by Node A.
    Assertion: Validation fails.
    """
    daq, uids, net, obs = palomar_setup
    
    # Node 0 handles module 1, Node 1 handles module 2
    # Make Node 1 also handle module 1
    daq.daq_nodes[1].module_ids.append(1)
    
    configs = {'obs': obs, 'daq': daq, 'network': net, 'uids': uids}
    validator = GlobalConfigValidator(configs)
    passed = validator.validate_all_rules()
    
    assert passed is False
    assert any("multiple" in t["info"].lower() or "assigned" in t["info"].lower() 
               for t in validator.report.tests if t["status"] == "ERROR")

def test_when_quabo_ip_duplicated_in_obs_then_rejected(palomar_setup):
    """
    Intent: Ensure each module in obs_config has a unique IP address.
    """
    daq, uids, net, obs = palomar_setup
    
    # Duplicate the first module's IP in the PTI dome module
    # Palomar setup has 4 domes, each with 1 module.
    obs.domes[1].modules[0].ip_addr = obs.domes[0].modules[0].ip_addr
    
    configs = {'obs': obs, 'daq': daq, 'network': net, 'uids': uids}
    validator = GlobalConfigValidator(configs)
    passed = validator.validate_all_rules()
    
    assert passed is False
    assert any("module id" in t["info"].lower() or "assigned" in t["info"].lower() 
               for t in validator.report.tests if t["status"] == "ERROR")
