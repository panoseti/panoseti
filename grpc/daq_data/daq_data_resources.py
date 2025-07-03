"""
Common functions for the gRPC DaqUtils service.
"""
import os
import json
import logging
import datetime
from typing import List, Callable, Tuple, Any
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
daq_data_config_file = 'daq_data_config.json'

# Configuration for hashpipe_io thread
default_hashpipe_io_thread_cfg = "default_hashpipe_io_thread_config.json"

with open(cfg_dir/default_hashpipe_io_thread_cfg) as f:
    default_hashpipe_io_cfg = json.load(f)


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
