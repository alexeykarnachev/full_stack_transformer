import logging
import sys
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)
FORMATTER = '[%(asctime)s %(module)s %(funcName)s %(levelname)s] %(message)s'


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Handler for unhandled exceptions that will write to the logs"""
    if issubclass(exc_type, KeyboardInterrupt):
        # call the default excepthook saved at __excepthook__
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


def get_rotating_file_handler(log_file: str, level: str, max_bytes: int = 10485760, backup_count: int = 5) -> Dict:
    handler_dict = {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': level,
        'formatter': 'default',
        'filename': log_file,
        'mode': 'a',
        'maxBytes': max_bytes,
        'backupCount': backup_count,
    }

    return handler_dict


def get_console_output_handler(level):
    handler_dict = {
        'class': 'logging.StreamHandler',
        'level': level,
        'formatter': 'default',
    }

    return handler_dict


def get_log_config(log_dir: Path) -> dict:
    log_dir.mkdir(exist_ok=True, parents=True)
    info_file = str(log_dir / 'info.log')
    errors_file = str(log_dir / 'errors.log')
    critical_file = str(log_dir / 'critical.log')
    debug_file = str(log_dir / 'debug.log')

    handlers = {
        'info_file': get_rotating_file_handler(info_file, 'INFO'),
        'debug_file': get_rotating_file_handler(debug_file, 'DEBUG'),
        'errors_file': get_rotating_file_handler(errors_file, 'ERROR'),
        'critical_file': get_rotating_file_handler(critical_file, 'CRITICAL'),
        'console': get_console_output_handler('INFO')
    }

    log_config = {
        'disable_existing_loggers': False,
        'version': 1,
        'formatters': {
            'default': {
                'format': FORMATTER
            }
        },
        'handlers': handlers,

        'loggers': {
            '': {
                'handlers': list(handlers.keys()),
                'level': 'DEBUG'
            }
        }
    }

    return log_config
