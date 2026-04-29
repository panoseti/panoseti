"""
State-aware test scheduler.
Groups tests into batches by required state and plans transitions.
"""

class StateAwareScheduler:
    def __init__(self, state_machine, tests_toml_path: str):
        pass

    def schedule(self, items):
        """Re-orders pytest items into optimized state batches."""
        return items
