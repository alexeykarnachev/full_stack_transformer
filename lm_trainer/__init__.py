import sys

from lm_trainer.utilities import log_config

sys.excepthook = log_config.handle_unhandled_exception

__version__ = '0.0.1'
