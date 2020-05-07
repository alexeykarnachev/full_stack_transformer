import sys

from full_stack_transformer.utilities import log_config

sys.excepthook = log_config.handle_unhandled_exception

__version__ = '0.0.3'
