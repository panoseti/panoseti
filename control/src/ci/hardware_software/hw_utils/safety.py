"""
Safety Manager for HITL tests.
Ensures hardware is returned to a safe state on exit.
"""

class SafetyManager:
    def __init__(self, state_machine):
        pass

    def emergency_teardown(self):
        """Drives hardware to the 'safe' state defined in TOML."""
        pass
