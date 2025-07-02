#!/usr/bin/env python3

"""
The Python implementation of a gRPC UbloxControl server.

Requires the following to function correctly:
    1. A POSIX-compliant operating system.
    2. A valid connection to a ZED-F9T u-blox chip.
    3. All Python packages specified in requirements.txt.
"""
import logging
import queue
import random
import sys
from concurrent import futures
import threading
from threading import Event, Thread
from queue import Queue
from serial import Serial
import time
import re
import urllib.parse

import grpc

# gRPC reflection service: allows clients to discover available RPCs
from grpc_reflection.v1alpha import reflection

# standard gRPC protobuf types + utility functions
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf import timestamp_pb2

from pyubx2 import POLL, UBX_PAYLOADS_POLL, UBX_PROTOCOL, UBXMessage, UBXReader

# protoc-generated marshalling / demarshalling code
import ublox_control_pb2
import ublox_control_pb2_grpc

# alias messages to improve readability
from ublox_control_resources import *
from init_f9t_tests import *  #run_all_tests, is_os_posix, check_client_f9t_cfg_keys


def f9t_io_data(
    device: str,
    timeout: float,
    baudrate: int,
    read_queues: List[Queue],
    read_queue_freemap: List[bool],
    send_queue: Queue,
    stop: Event,
    valid: Event,
    logger: logging.Logger,
    **kwargs
):
    """
    THREADED
    Read and parse inbound UBX data and place
    raw and parsed data on queue.

    Send any queued outbound messages to receiver.
    :license: BSD 3-Clause
    """
    logger.info(f"Created a new f9t_io thread with the following options: {kwargs=}")
    valid.clear()  # indicate f9t io channel is currently invalid
    try:
        with Serial(device, baudrate, timeout=timeout) as stream:
            ubr = UBXReader(stream, protfilter=UBX_PROTOCOL)
            logger.info("f9t_io thread connected to the F9t chip. The f9t io channel is now valid")
            valid.set()
            while not stop.is_set():
                (raw_data, parsed_data) = ubr.read()
                if parsed_data:
                    for read_queue, is_allocated in zip(read_queues, read_queue_freemap):
                        if is_allocated:  # only populate read_queues that are actively being used
                            read_queue.put((raw_data, parsed_data))

                # refine this if outbound message rates exceed inbound
                while not send_queue.empty():
                    data = send_queue.get(False)
                    if data is not None:
                        ubr.datastream.write(data.serialize())
                    send_queue.task_done()
    except Exception as err:
        logger.critical(f"f9t_io thread encountered a fatal exception! {err}")
        raise err
    finally:
        valid.clear()
        logger.info("f9t_io thread exited")
    # print("f9t_io_data exited.")


def f9t_io_data_DEBUG(
        device: str,
        timeout: float,
        baudrate: int,
        read_queues: List[Queue],
        read_queue_freemap: List[bool],
        send_queue: Queue,
        stop: Event,
        valid: Event,
        logger: logging.Logger,
        **kwargs
):
    """
    THREADED
    Read and parse inbound UBX data and place
    raw and parsed data on queue.

    Send any queued outbound messages to receiver.
    :license: BSD 3-Clause
    """
    logger.info(f"Created a new DEBUG f9t_io thread with the following options: {kwargs=}")
    valid.clear()  # indicate f9t io channel is currently invalid
    try:
        if "early_exit_delay_seconds" in kwargs:
            early_exit_counter = kwargs["early_exit_delay_seconds"]
        else:
            early_exit_counter = 30
        # simulate successfully opening a connection to the target f9t chip
        logger.info("f9t_io thread connected to the target F9t chip. The f9t io channel is now valid")
        valid.set()
        while not stop.is_set():
            time.sleep(1)
            parsed_data = {
                "identity": "TEST",
                "test_field": random.choice(["fox", "socks", "box", "knox"]),
                "id": random.randint(-100000, 100000)
            }
            raw_data = random.randint(-100, 100)
            # logger.debug(f"[f9t_io -> read_queues] {raw_data=}, {parsed_data=}")
            # (raw_data, parsed_data) = random.randint(-100, 100), parsed_data
            if parsed_data:
                for read_queue, is_allocated in zip(read_queues, read_queue_freemap):
                    if is_allocated:  # only populate read_queues that are actively being used
                        read_queue.put((raw_data, parsed_data))

            # refine this if outbound message rates exceed inbound
            while not send_queue.empty() and not stop.is_set():
                data = send_queue.get(False)
                if data is not None:
                    ...
                    # ubr.datastream.write(data.serialize())
                    # logger.debug(f"[send_queue -> f9t_io] {data=}")
                send_queue.task_done()
            if "early_exit" in kwargs and kwargs["early_exit"]:
                early_exit_counter -= 1
                if early_exit_counter == 0:
                    raise TimeoutError("test f9t_io thread unexpected termination")
    except Exception as err:
        logger.critical(f"f9t_io thread encountered a fatal exception! {err}")
        raise err
    finally:
        valid.clear()
        logger.info("f9t_io thread exited")


"""gRPC server implementing UbloxControl RPCs"""
class UbloxControlServicer(ublox_control_pb2_grpc.UbloxControlServicer):
    """Provides methods that implement functionality of an u-blox control server."""

    def __init__(self, server_cfg):
        # verify the server is running on a POSIX-compliant system
        test_result, msg = is_os_posix()
        assert test_result, msg

        # Initialize mesa monitor for synchronizing access to the F9T chip
        #   "Writers" = threads executing the InitF9t RPC.
        #   "Readers" = threads executing any other UbloxControl RPC that depends on F9t data
        self._f9t_rw_lock_state = {
            "wr": 0,  # waiting readers
            "ww": 0,  # waiting writers
            "ar": 0,  # active readers
            "aw": 0,  # active writers
        }
        self._f9t_lock = threading.Lock()
        self._read_ok_condvar = threading.Condition(self._f9t_lock)
        self._write_ok_condvar = threading.Condition(self._f9t_lock)
        self._active_clients = {}  # dict of tid : context.peer() for debugging

        self._server_cfg = server_cfg

        # Create the server's logger
        self.logger = make_rich_logger(__name__, level=logging.DEBUG)

        # Load default F9t configuration
        with open(cfg_dir/self._server_cfg["InitF9t"]["default_f9t_cfg_file"], "r") as f:
            self._f9t_cfg = json.load(f)

        ## State for single producer, multiple consumer F9t access
        # A single IO thread manages the dataflow between multiple concurrent RPC threads and the F9t:
        #   [single RPC writer -> F9t IO thread] write POLL and SET requests to the F9t
        #   [F9t IO thread -> many RPC readers] read (GET) packets from the F9t and write to max_workers read_queues
        self._f9t_io_thread: Thread = None  # initialized by

        # Create an array of read_queues and freemap locks to support up to max_worker concurrent reader RPCs
        self._read_queues = []  # Duplicate queues to implement single producer, multiple independent consumer model
        self._read_queues_freemap = []  # True iff corresponding queue is allocated to a reader
        for _ in range(server_cfg['max_workers']):
            self._read_queues.append(Queue(maxsize=server_cfg['max_read_queue_size']))
            self._read_queues_freemap.append(False)
        self._send_queue = Queue()  # Used by InitF9t and PollMessage to send SET and POLL requests to the F9t chip
        self._f9t_io_stop = Event()  # Signals F9t IO thread to release the serial connection to the F9t
        self._f9t_io_valid = Event()  # Asserted only if f9t_io thread is running

        # Start F9t io thread if server_cfg points to a valid f9t_cfg and doesn't require an initial InitF9t RPC
        if self._server_cfg["InitF9t"]["allow_init_from_default"] and self._f9t_cfg["is_valid"]:
            self.logger.info(f"Creating the initial f9t_io thread from config: "
                             f"{self._server_cfg["InitF9t"]["allow_init_from_default"]=} and "
                             f"{self._f9t_cfg["is_valid"]=}.")
            self._server_cfg['f9t_init_valid'] = True
            self._start_f9t_io_thread(self._f9t_cfg)
        else:
            self.logger.warning(f"An InitF9t call is required to start the f9t_io thread: "
                                f"{self._server_cfg["InitF9t"]["allow_init_from_default"]=} and "
                                f"{self._f9t_cfg["is_valid"]=}.")

            self._server_cfg['f9t_init_valid'] = False

    def __del__(self):
        """Terminate f9t_io thread and free its F9t serial connection"""
        self._server_cfg['f9t_init_valid'] = False
        all_ok = True
        all_ok &= self._stop_f9t_io_thread()
        # # wake threads waiting to acquire the lock
        # with self._f9t_lock:
        #     self._read_ok_condvar.notify_all()
        #     self._write_ok_condvar.notify_all()

        # check if state was updated properly
        for thread_state, num_threads in self._f9t_rw_lock_state.items():
            if num_threads != 0:
                self.logger.critical(f"[rw lock] unexpected threads in state {thread_state} at termination!\n"
                                     f"{self._f9t_rw_lock_state=}")
                all_ok &= False
        if all_ok:
            self.logger.info("Successfully released all resources")
        else:
            self.logger.critical("Some server resources were not released")

    @contextmanager
    def _f9t_lock_writer(self, context):
        active = False
        try:
            with self._f9t_lock:
                # BEGIN check-in critical section
                # All reader RPCs are long-lived server streaming operations.
                # The server's synchronization logic will prevent writes to F9t state while any reader RPCs are active,
                # so we should cancel any writer RPCs immediately
                if self._f9t_rw_lock_state['ar'] > 0:
                    active_clients = str(list(self._active_clients.values()))
                    # print(active_clients)
                    emsg = (f"Cannot modify F9t state because there are {self._f9t_rw_lock_state['ar']} active "
                            f"CaptureUblox clients. Stop these client processes then try again: {active_clients=}.")
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
                # Wait until no active readers or active writers
                self.logger.debug(f"(writer) check-in (start):\t{self._f9t_rw_lock_state=}")
                while context.is_active() and (self._f9t_rw_lock_state['aw'] + self._f9t_rw_lock_state['ar']) > 0:
                    self._f9t_rw_lock_state['ww'] += 1
                    self._write_ok_condvar.wait(timeout=5)
                    self._f9t_rw_lock_state['ww'] -= 1

                # check if environment is still valid
                if not context.is_active():
                    emsg = "client cancelled rpc during writer lock acquisition (skipping to check-out)"
                    self.logger.warning(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                if context.is_active() and self._server_cfg['f9t_init_valid'] and not self._is_f9t_io_valid():
                    emsg = (f"The f9t_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['f9t_init_valid'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)
                # activate the writer
                self._f9t_rw_lock_state['aw'] += 1
                active = True
                self._active_clients[threading.get_ident()] = urllib.parse.unquote(context.peer())
                self.logger.debug(f"(writer) check-in (end):\t\t{self._f9t_rw_lock_state=}")
                # END check-in critical section
            yield None
        except RuntimeError as err:
            pass
        finally:
            with self._f9t_lock:
                # BEGIN check-out critical section
                self.logger.debug(f"(writer) check-out (start):\t{self._f9t_rw_lock_state=}")
                if active:  # handle edge cases where thread is interrupted or has an error during lock acquire
                    self._f9t_rw_lock_state['aw'] = self._f9t_rw_lock_state['aw'] - 1  # no longer active
                    del self._active_clients[threading.get_ident()]
                # Wake up waiting readers or a waiting writer (prioritize waiting writers).
                if self._f9t_rw_lock_state['ww'] > 0:  # Give lock priority to waiting writers
                    self._write_ok_condvar.notify()
                elif self._f9t_rw_lock_state['wr'] > 0:
                    self._read_ok_condvar.notify_all()
                self.logger.debug(f"(writer) check-out (end):\t{self._f9t_rw_lock_state=}")
                # END check-out critical section

    @contextmanager
    def _f9t_lock_reader(self, context):
        read_fmap_idx = -1  # remember which read_queue freemap entry corresponds to this thread
        active = False
        try:
            with self._f9t_lock:
                # BEGIN check-in critical section
                self.logger.debug(f"(reader) check-in (start):\t{self._f9t_rw_lock_state=}"
                                  f"\n{self._read_queues_freemap=}")
                # Wait until no active writers or waiting writers
                while context.is_active() and (self._f9t_rw_lock_state['aw'] + self._f9t_rw_lock_state['ww']) > 0:
                    self._f9t_rw_lock_state['wr'] += 1
                    self._read_ok_condvar.wait()
                    self._f9t_rw_lock_state['wr'] -= 1

                if not context.is_active():
                    emsg = "client context terminated during reader lock acquisition [skipping to check-out]"
                    self.logger.error(emsg)
                    context.cancel()

                # allocate a read queue for this thread
                for idx, is_allocated in enumerate(self._read_queues_freemap):
                    if not is_allocated:
                        read_fmap_idx = idx
                        self._read_queues_freemap[idx] = True
                        break

                # check if the allocation succeeded
                if read_fmap_idx < 0:
                    emsg = "_read_queues_freemap allocation failed during reader check-in! [SHOULD NEVER HAPPEN]"
                    self.logger.critical(emsg)
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # check if f9t_io is valid
                if context.is_active() and self._server_cfg['f9t_init_valid'] and not self._is_f9t_io_valid():
                    emsg = (f"the f9t_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['f9t_init_valid'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # activate the reader
                self._f9t_rw_lock_state['ar'] += 1
                active = True
                self._active_clients[threading.get_ident()] = urllib.parse.unquote(context.peer())
                self.logger.debug(f"(reader) check-in (end):\t\t{self._f9t_rw_lock_state=}, fmap_idx={read_fmap_idx}"
                                  f"\n{self._read_queues_freemap=}")
                # END check-in critical section
            yield read_fmap_idx
        # except RuntimeError as err:
        finally:
            with self._f9t_lock:
                # BEGIN check-out critical section
                self.logger.debug(f"(reader) check-out (start):\t{self._f9t_rw_lock_state=}")
                if active:
                    self._f9t_rw_lock_state['ar'] = self._f9t_rw_lock_state['ar'] - 1  # no longer active
                    del self._active_clients[threading.get_ident()]
                    self._read_queues_freemap[read_fmap_idx] = False  # release the read queue
                # Wake up waiting readers or a waiting writer (prioritize waiting writers).
                if self._f9t_rw_lock_state['ar'] == 0 and self._f9t_rw_lock_state['ww'] > 0:
                    self._write_ok_condvar.notify()
                elif self._f9t_rw_lock_state['wr'] > 0:
                    self._read_ok_condvar.notify_all()
                self.logger.debug(f"(reader) check-out (end):\t\t{self._f9t_rw_lock_state=}")
                # END check-out critical section

    def _start_f9t_io_thread(self, f9t_cfg):
        """Creates a new f9t io thread with the given f9t_cfg.
        @return: True iff the f9t_io thread was created and established a valid connection to the target F9t chip
        """
        # Terminate any currently alive f9t_io thread
        self._stop_f9t_io_thread()  # no effect if a f9t_io thread is not alive

        # Create new f9t_io_thread using the client's configuration
        self._f9t_io_stop.clear()
        self._f9t_io_thread = Thread(
            # target=f9t_io_data,
            target=f9t_io_data_DEBUG,
            args=(
                f9t_cfg['device'],
                f9t_cfg['timeout'],
                F9T_BAUDRATE,
                self._read_queues,
                self._read_queues_freemap,
                self._send_queue,
                self._f9t_io_stop,
                self._f9t_io_valid,
                self.logger,
            ),
            kwargs={
                "early_exit": False,  # causes the f9t_io thread have a fatal exception after the given delay
                "early_exit_delay_seconds": 25
            },
            daemon=False,
        )
        self._f9t_io_thread.start()

        # check if thread could be properly initialized
        self._f9t_io_valid.wait(1)
        if self._is_f9t_io_valid():
            self.logger.info("f9t_io thread alive and valid")
            return True
        else:
            self._stop_f9t_io_thread()
            return False

    def _is_f9t_io_valid(self):
        if self._f9t_io_thread is not None and self._f9t_io_thread.is_alive() and self._f9t_io_valid.is_set():
            return True
        elif self._f9t_io_thread is None:
            self.logger.warning("f9t_io thread is uninitialized")
        elif not self._f9t_io_thread.is_alive():
            self.logger.critical("f9t_io thread is not alive")
        elif not self._f9t_io_valid.is_set():
            self.logger.warning("f9t_io thread is alive but not valid")
        else:
            emsg = (f"unhandled is_f9t_io_valid case: "
                    f"{self._f9t_io_thread=}, "
                    f"{self._f9t_io_thread.is_alive()=},"
                    f"{self._f9t_io_valid=}")
            self.logger.critical(emsg)
            raise RuntimeError(emsg)  # SHOULD NEVER REACH HERE
        return False


    def _stop_f9t_io_thread(self):
        """Stops the f9t io thread. Idempotent behavior.
        @return: True iff the f9t io thread is not alive."""
        self._f9t_io_stop.set()  # signal f9t_io thread to exit gracefully
        if self._f9t_io_thread is not None and self._f9t_io_thread.is_alive():
            try:
                self._f9t_io_thread.join(5)  # wait until f9t_io exits
            except RuntimeError as rerr:
                self.logger.critical(f"encountered runtime error while stopping f9t_io thread: {rerr}")
            finally:
                if self._f9t_io_thread.is_alive():  # check if join succeeded or timeout happened while waiting
                    self.logger.critical(f"Could not stop f9t_io thread")
                    return False
                else:
                    self.logger.info(f"Successfully terminated f9t_io thread")
                    return True
        else:
            self.logger.debug("no f9t_io thread to stop (doing nothing)")
            return True

    def InitF9t(self, request, context):
        """Updates the F9t configuration in a transaction.
        The transaction is cancelled if any of the validation tests fail.
            1. Validate a client-supplied F9t configuration.
            2. Start a new f9t_io thread and verify its ability to send commands and receive data packets.
            3. Commit the client's configuration to internal server state.
        [f9t writer]
        """
        self.logger.info(f"new InitF9t rpc from {urllib.parse.unquote(context.peer())}")
        f9t_cfg_keys_to_copy = ['device', 'chip_name', 'chip_uid', 'timeout', 'cfg_key_settings', 'comments']
        test_results = []

        client_f9t_cfg = MessageToDict(request.f9t_cfg)
        # TODO: Validate f9t_cfg here:
        #  0. System is POSIX [done]
        #  1. all keys in f9t_cfg_keys_to_copy are present in client_f9t_cfg [done]
        #  2. device file is valid [done]
        #  3. device file points to an f9t chip
        #  4. we can send SET and POLL requests to the chip and read responses with GET
        #  5. all keys under "set_cfg_keys" are valid and supported by pyubx2
        cfg_all_pass, cfg_test_results = run_all_tests(
            test_fn_list=[
                is_os_posix,
                check_client_f9t_cfg_keys,
                # is_device_valid,
            ],
            args_list=[
                [],
                [f9t_cfg_keys_to_copy, client_f9t_cfg.keys()],
                # [client_f9t_cfg['device']]
            ]
        )
        commit_changes = cfg_all_pass
        test_results.extend(cfg_test_results)

        # TODO: Do F9t initialization
        #   Set the configuration according to client_f9t_cfg['set_cfg_keys'].
        #   NOTE: unspecified keys are returned to default values.
        if commit_changes:
            # Only enter critical section as a writer if the client's F9t configuration is valid
            self.logger.info("Client f9t_cfg is valid")
            with self._f9t_lock_writer(context):
                try:
                    # BEGIN critical section for F9tInit [write] access
                    time.sleep(0.5)  # DEBUG: add delay to expose race conditions
                    self._start_f9t_io_thread(client_f9t_cfg)

                    # Run tests with the f9t_io thread to verify f9t initialization succeeded
                    # TODO: remove these debug tests
                    def always_fail():
                        return False, "fail"
                    def always_pass():
                        return True, "pass"

                    init_all_pass, init_test_results = run_all_tests(
                        test_fn_list=[
                            # check_f9t_dataflow # TODO: implement this
                            poll_nav_messages,
                            # always_fail()
                            always_pass
                        ],
                        args_list=[
                            # [client_f9t_cfg],
                            [self._send_queue],
                            []
                        ]
                    )
                    commit_changes = init_all_pass
                    test_results.extend(init_test_results)
                    # Commit changes to self._f9t_cfg only if all tests pass
                    if commit_changes and context.is_active():
                        self.logger.debug("f9t_io thread created with client f9t_cfg passes all tests.")
                        self.logger.info(f"InitF9t transaction succeeded. New F9t CFG: {self._f9t_cfg=}")
                        self._server_cfg['f9t_init_valid'] = True
                        self._f9t_cfg['is_valid'] = True
                        for key in f9t_cfg_keys_to_copy:
                            # self.logger.info(f"F9T CFG Update: f9t_cfg['{key}'] = {repr(client_f9t_cfg[key])}")
                            self._f9t_cfg[key] = client_f9t_cfg[key]
                        message = "InitF9t transaction successful"
                        init_status = ublox_control_pb2.InitF9tResponse.InitStatus.SUCCESS
                finally:
                    # If any tests fail, or we encounter any exceptions in the critical section, do the following:
                    #   Cancel the transaction and rollback state to old initialization:
                    if not commit_changes or not context.is_active():
                        self.logger.warning(f"InitF9t transaction failed.")
                        if self._server_cfg['f9t_init_valid'] and self._f9t_cfg['is_valid']:
                            # Restart the f9t_io thread with the old configuration only if old config was valid
                            self.logger.info(f"Restarting the f9t_io thread from previous valid F9t CFG: {self._f9t_cfg=}")
                            if not self._start_f9t_io_thread(self._f9t_cfg):
                                self.logger.critical("Failed to restart valid f9t_io thread after cancelled InitF9t transaction")
                        if context.is_active():
                            emsg = ("InitF9t transaction cancelled (some f9t_io tests failed). "
                                       "See the test_cases field for information about failing tests")
                            context.abort(grpc.StatusCode.ABORTED, emsg)
                        else:
                            context.cancel()
                    # END critical section for F9t [write] access
        else:
            emsg = ("InitF9t transaction cancelled (invalid client_f9t_cfg). "
                       "See the test_cases field for information about failing tests")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, emsg)

        # Send summary of initialization process to client
        init_f9t_response = ublox_control_pb2.InitF9tResponse(
            init_status=init_status,
            message=message,
            f9t_cfg=ParseDict(self._f9t_cfg, Struct()),
            test_results=test_results,
        )
        return init_f9t_response

    def CaptureUblox(self, request, context):
        """Forward u-blox packets to the client. [reader]"""
        # unpack the requested message pattern filters
        self.logger.info(f"new CaptureUblox rpc from {urllib.parse.unquote(context.peer())}")
        patterns = request.patterns
        regex_list = [re.compile(pattern) for pattern in patterns]
        # TODO: check if the patterns are valid
        with (self._f9t_lock_reader(context) as rid):  # rid = allocated reader id
            # BEGIN critical section for F9t [read] access
            # Clear the read_queue of old data
            rq = self._read_queues[rid]
            while not rq.empty():
                rq.get()
            if self._server_cfg['f9t_init_valid'] and self._is_f9t_io_valid():
                # self.logger.info("Streaming messages")
                while context.is_active() and self._is_f9t_io_valid():
                    # self.logger.debug("waiting for input")
                    try:
                        # wait for next packet from f9t_io thread
                        # add a timeout of 30s to avoid starvation in case the f9t_io thread unexpectedly exits
                        # while this thread is blocking on the read_queue
                        raw_data, parsed_data = rq.get(timeout=10)
                        #self.logger.debug(f"got {parsed_data=}")
                        # pack data into a CaptureUbloxResponse message
                        # TODO: replace these hard-coded values with packets received from the connected u-blox chip
                        name = parsed_data['identity']  #name = parsed_data.identity
                        # TODO: find or create handlers to unpack each packet type into a dictionary of KV pairs
                        parsed_data_dict = {
                            'test_field': parsed_data['test_field'],
                            'uid': parsed_data['id'],
                        }
                        timestamp = timestamp_pb2.Timestamp()
                        timestamp.GetCurrentTime()

                        gnss_packet = ublox_control_pb2.CaptureUbloxResponse(
                            type=ublox_control_pb2.CaptureUbloxResponse.Type.DATA,
                            name=name,
                            parsed_data=ParseDict(parsed_data_dict, Struct()),
                            timestamp=timestamp
                        )
                        # send packet if its name matches any of the given patterns or if no patterns were given.
                        if len(regex_list) == 0 or any(regex.search(name) for regex in regex_list):
                            yield gnss_packet
                    except queue.Empty:
                        self.logger.warning("f9t_io thread may have stopped sending data")
                        continue
                # log reason why streaming stopped
                if not context.is_active():
                    self.logger.info(f"CaptureUblox client disconnected")
                else:
                    emsg = (f"The f9t_io thread data stream unexpectedly became invalid! "
                            f"Check the UbloxControl server logs to debug this issue")
                    self.logger.critical(emsg)
                    context.abort(grpc.StatusCode.INTERNAL, emsg)
            else:
                emsg = "Uninitialized F9T. Run InitF9t with a valid f9t configuration to initialize it."
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            # END critical section for F9t [read] access

def serve(server_cfg):
    """Create the gRPC server threadpool and start providing the UbloxControl service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_cfg['max_workers']))
    ublox_control_pb2_grpc.add_UbloxControlServicer_to_server(
        UbloxControlServicer(server_cfg), server
    )

    # Add RPC reflection to show available commands to users
    SERVICE_NAMES = (
        ublox_control_pb2.DESCRIPTOR.services_by_name["UbloxControl"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start gRPC and configure to listen on port 50051
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"The gRPC services {SERVICE_NAMES} are running.\nEnter CTRL+C to stop them.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        grace = server_cfg["shutdown_grace_period"]
        print(f"'^C' received, shutting down the server in {grace} seconds.")
        server.stop(grace=grace).wait(grace)
        sys.exit(0)



if __name__ == "__main__":
    # Load server configuration
    server_cfg_file = "ublox_control_server_config.json"
    with open(cfg_dir / server_cfg_file, "r") as f:
        server_cfg = json.load(f)
    serve(server_cfg)
