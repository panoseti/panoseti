"""
Hardware Topology adapter.
Loads observatory and DAQ configs to gate tests by available hardware.
"""

class HwTopology:
    def __init__(self, config_dir: str):
        pass

    def gate(self, test_id: str) -> bool:
        """Returns True if the current topology supports the given test."""
        return True

    def capabilities(self) -> set[str]:
        return set()
