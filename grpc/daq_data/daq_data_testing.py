"""
Test cases and test utilities for the gRPC DaqData service.

Each function must have type Callable[..., Tuple[bool, str]] as the example below:

    def is_even(n: int) -> Tuple[bool, str]:
        if n % 2 == 0:
            return True, f"{n} is even"
        else:
            return False, f"{n} is odd"
"""
import os
import datetime
import redis
from pathlib import Path
from typing import List, Callable, Tuple, Any, Dict

from rich import print

from daq_data_resources import make_rich_logger
from daq_data_pb2 import StreamImagesResponse, StreamImagesRequest

""" Testing utils """

def run_all_tests(
        test_fn_list: List[Callable[[Any], Tuple[bool, str]]],
        args_list: List[List[Any]],
) -> tuple[bool, list[Any]]:
    """
    Runs each test function in [test_functions].
    To ensure correct behavior new test functions have type Callable[..., Tuple[bool, str]] to ensure correct behavior.
    Returns enum init_status and a list of test_results.
    """
    assert len(test_fn_list) == len(args_list), "test_fn_list must have the same length as args_list"
    def get_test_name(test_fn) -> str:
        return f"{test_fn.__module__}.{test_fn.__name__}"

    all_pass = True
    test_results = []
    for test_fn, args in zip(test_fn_list, args_list):
        test_result, message = test_fn(*args)
        if not test_result:
            all_pass &= False
        test_result = {
            "name": get_test_name(test_fn),
            "result": test_result,
            "message": message
        }
        test_results.append(test_result)
    return all_pass, test_results


""" Test cases """
def test_redis_connection(host, port=6379, socket_timeout=1, logger=None) -> Tuple[bool, str]:
    """
    Test Redis connection with specified connection parameters.
        1. Connect to Redis.
        2. Perform a series of pipelined write operations to a test hashset.
        3. Check if all writes were successful.
    """
    failures = 0

    try:
        # print(f"Connecting to {host}:{port}")
        if logger: logger.debug(f"Connecting to {host}:{port}")
        r = redis.Redis(host=host, port=port, db=0, socket_timeout=socket_timeout)
        if not r.ping():
            # raise FileNotFoundError(f'Cannot connect to {host}:{port}')
            if logger: logger.error(f"Cannot connect to {host}:{port}")
            return False, f'Cannot connect to {host}:{port}'

        timestamp = datetime.datetime.now().isoformat()
        # Create a redis pipeline to efficiently send key updates.
        pipe = r.pipeline()

        # Queue updates to a test hash: write current timestamp to 10 test keys
        for i in range(20):
            field = f't{i}'
            value = datetime.datetime.now().isoformat()
            pipe.hset('TEST', field, value)

        # Execute the pipeline and get results
        results = pipe.execute(raise_on_error=False)

        # Check if each operation succeeded
        success = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                success.append('0')
                failures += 1
                print(f"Command {i} failed: {result=}")
                if logger: logger.debug(f"Command {i} failed: {result=}")
            else:
                success.append('1')
        # print(f'[{timestamp}]: success = [{" ".join(success)}]')
        if logger:
            if all(success):
                logger.debug(f'success = [{" ".join(success)}]')

    except Exception as e:
        # Fail safely by reporting a failure in case of any exceptions
        return False, f"Error: {e}"
    test_result = (failures == 0)
    return test_result, f"{failures=}"

""" Server config verification """

def check_client_cfg_keys(required_keys: List[str], client_keys: List[str]) -> Tuple[bool, str]:
    """Verify the client's f9t config keys contain all required keys."""
    required_key_set = set(required_keys)
    client_key_set = set(client_keys)
    if set(required_key_set).issubset(client_key_set):
        return True, "all required f9t_cfg keys are present"
    else:
        required_key_diff = required_key_set.difference(client_key_set)
        return False, f"the given f9t_cfg is missing the following required keys: {required_key_diff}"


def is_hashpipe_running() -> Tuple[bool, str]:
    """TODO: check whether hashpipe is running."""
    is_running = False # TODO: add real checks
    if is_running:
        return False, "hashpipe is not running"
    return True, "hashpipe is running"


def is_os_posix():
    """Verify the server is running in a POSIX environment."""
    if os.name == 'posix':
        return True, f"detected a POSIX-compliant system"
    else:
        return False, f"{os.name} is not supported yet"

def check_hashpipe_io_dataflow(read_queue) -> Tuple[bool, str]:
    """TODO:"""

    return True, ""

