"""
Common functions for the gRPC DaqUtils service.
"""
import os
import json
import logging
import datetime
from typing import List, Callable, Tuple, Any, Dict
from contextlib import contextmanager
from pathlib import Path
import redis

from rich import print
from rich.logging import RichHandler
from rich.pretty import pprint

# from unittest import TestResult
import daq_data_pb2
from daq_data_pb2 import TestCase, CaptureScienceResponse, CaptureScienceRequest


""" Config globals"""
cfg_dir = Path('config')
default_hp_io_thread_config_file = 'default_hp_io_config.json'

# Configuration
with open(cfg_dir/default_hp_io_thread_config_file) as f:
    default_hp_io_thread_config = json.load(f)


def make_rich_logger(name, level=logging.DEBUG):
    LOG_FORMAT = (
        "[tid=%(thread)d] [%(funcName)s()] %(message)s "
    )

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        # datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(name)
