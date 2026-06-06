import logging
from src.logging_setup import configure_logging

def test_configure_logging_sets_level_and_is_idempotent():
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    n_handlers = len(root.handlers)
    configure_logging("INFO")            # second call must not stack handlers
    assert len(root.handlers) == n_handlers
    assert root.level == logging.INFO
