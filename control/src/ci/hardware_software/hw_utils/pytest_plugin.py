"""
Pytest plugin for HITL testing.
Handles test collection, classification, and batch reporting.
"""


def pytest_collection_modifyitems(config, items):
    """Classify tests and skip those with unmet topology requirements."""
    pass

def pytest_runtest_protocol(item, nextitem):
    """Intercept to ensure scheduler batching and transitions."""
    pass

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the HITL batch plan vs actual execution summary."""
    pass
