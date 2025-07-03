#!/usr/bin/env python3

"""
The Python implementation of a gRPC DaqData server.

Requires following to function correctly:
    1. A POSIX-compliant operating system.
    2. All Python packages specified in requirements.txt.
    3. A connection to a panoseti module.
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
import numpy as np

import grpc

# gRPC reflection service: allows clients to discover available RPCs
from grpc_reflection.v1alpha import reflection

# standard gRPC protobuf types + utility functions
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf import timestamp_pb2

# protoc-generated marshalling / demarshalling code
import daq_data_pb2
import daq_data_pb2_grpc
from daq_data_pb2 import PanoImage, TestCase, CaptureScienceResponse, CaptureScienceRequest

from daq_data_resources import *
from daq_data_testing import *


def hp_io_data_DEBUG(
        named_pipe_path: Path,
        timeout: float,
        read_queues: List[Queue],
        read_queue_freemap: List[bool],
        # send_queue: Queue,
        stop_io: Event,
        valid: Event,
        logger: logging.Logger,
        **kwargs
):
    """ Receive pulse-height and movie-mode data from hashpipe and broadcast it to all activeread queues. """
    logger.info(f"Created a new DEBUG hp_io thread with the following options: {kwargs=}")
    valid.clear()  # indicate hashpipe io channel is currently invalid
    try:
        if "early_exit_delay_seconds" in kwargs:
            early_exit_counter = kwargs["early_exit_delay_seconds"]
        else:
            early_exit_counter = 30
        """TODO: implement this io with named pipes"""
        valid.set()
        while not stop_io.is_set():
            time.sleep(1)
            # TODO: don't hardcode and get values from hashpipe in someway..
            header = {"test0": 0, "test1": 1}
            image_array = np.random.randint(low=0, high=2**16, size=[32,32])

            parsed_data = {
                "header": header,
                "image_array": image_array,
                "type": PanoImage.Type.MOVIE,
            }
            if parsed_data:
                for read_queue, is_allocated in zip(read_queues, read_queue_freemap):
                    if is_allocated:  # only populate read_queues that are actively being used
                        read_queue.put(parsed_data)

            if "early_exit" in kwargs and kwargs["early_exit"]:
                early_exit_counter -= 1
                if early_exit_counter == 0:
                    raise TimeoutError("test hp_io thread unexpected termination")
    except Exception as err:
        logger.critical(f"hp_io thread encountered a fatal exception! {err}")
        raise err
    finally:
        valid.clear()
        logger.info("hp_io thread exited")


"""gRPC server implementing DaqData RPCs"""
class DaqDataServicer(daq_data_pb2_grpc.DaqDataServicer):
    """Provides methods that implement functionality of an u-blox control server."""

    def __init__(self, server_cfg):
        # verify the server is running on a POSIX-compliant system
        test_result, msg = is_os_posix()
        assert test_result, msg

        # Initialize mesa monitor for synchronizing access to the hp_io thread
        #   "Writers" = threads changing server state
        #   "Readers" = all other threads
        self._rw_lock_state = {
            "wr": 0,  # waiting readers
            "ww": 0,  # waiting writers
            "ar": 0,  # active readers
            "aw": 0,  # active writers
        }
        self._hp_io_lock = threading.Lock()
        self._read_ok_condvar = threading.Condition(self._hp_io_lock)
        self._write_ok_condvar = threading.Condition(self._hp_io_lock)
        self._active_clients = {}  # dict of tid : context.peer() for debugging

        self._server_cfg = server_cfg

        # Create the server's logger
        self.logger = make_rich_logger(__name__, level=logging.DEBUG)

        # Load default hahspipe_io configuration
        with open(cfg_dir/self._server_cfg["default_hp_io_config_file"], "r") as f:
            self._hp_io_cfg = json.load(f)

        ## State for single producer, multiple consumer hp_io access
        # A single IO thread manages the dataflow between multiple concurrent RPC threads and the hp_io thread:
        #   [single RPC writer -> hp_io thread] send messages to the hp_io thread
        #   [hp_io thread -> many RPC readers] broadcast image data to active read_queues
        self._hp_io_thread: Thread = None

        # Create an array of read_queues and freemap locks to support up to max_worker concurrent reader RPCs
        self._read_queues = []  # Duplicate queues to implement single producer, multiple independent consumer model
        self._read_queues_freemap = []  # True iff corresponding queue is allocated to a reader
        for _ in range(server_cfg['max_workers']):
            self._read_queues.append(Queue(maxsize=server_cfg['max_read_queue_size']))
            self._read_queues_freemap.append(False)
        # self._send_queue = Queue()  # Used
        self._stop_io = Event()  # Signals hp_io thread to exit
        self._hp_io_valid = Event()  # Set only if the hp_io thread is active and collecting data

        # Start the hp_io thread if server_cfg points to a valid hp_io_cfg
        if self._server_cfg["allow_init_from_default"] and self._hp_io_cfg["is_valid"]:
            self.logger.info(f"Creating the initial hp_io thread from config: "
                             f"{self._server_cfg["allow_init_from_default"]=} and "
                             f"{self._hp_io_cfg["is_valid"]=}.")
            self._server_cfg['hp_io_init'] = True
            self._start_hp_io_thread(self._hp_io_cfg)
        else:
            self.logger.warning(f"An InitHpIo call is required to start the hp_io thread: "
                                f"{self._server_cfg["allow_init_from_default"]=} and "
                                f"{self._hp_io_cfg["is_valid"]=}.")

            self._server_cfg['hp_io_init'] = False

    def __del__(self):
        """
        Cleanup resources:
            1. hp_io thread
        """
        self._server_cfg['hp_io_init'] = False
        all_ok = True
        all_ok &= self._stop_hp_io_thread()

        # check if state was updated properly
        for thread_state, num_threads in self._rw_lock_state.items():
            if num_threads != 0:
                self.logger.critical(f"[rw lock] unexpected threads in state {thread_state} at termination!\n"
                                     f"{self._rw_lock_state=}")
                all_ok &= False
        if all_ok:
            self.logger.info("Successfully released all resources")
        else:
            self.logger.critical("Some server resources were not released")

    @contextmanager
    def _rw_lock_writer(self, context):
        tid = threading.get_ident()
        active = False
        try:
            with self._hp_io_lock:
                # BEGIN check-in critical section
                # All reader RPCs are long-lived server streaming operations.
                # The server's synchronization logic will prevent writes to F9t state while any reader RPCs are active,
                # so we should cancel any writer RPCs immediately
                if self._rw_lock_state['ar'] > 0:
                    active_clients = str(list(self._active_clients.values()))
                    # print(active_clients)
                    emsg = (f"Cannot modify F9t state because there are {self._rw_lock_state['ar']} active "
                            f"CaptureScience clients. Stop these client processes then try again: {active_clients=}.")
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
                # Wait until no active readers or active writers
                self.logger.debug(f"(writer) check-in (start):\t{self._rw_lock_state=}")
                while context.is_active() and (self._rw_lock_state['aw'] + self._rw_lock_state['ar']) > 0:
                    self._rw_lock_state['ww'] += 1
                    self._write_ok_condvar.wait(timeout=5)
                    self._rw_lock_state['ww'] -= 1

                # check if environment is still valid
                if not context.is_active():
                    emsg = "client cancelled rpc during writer lock acquisition (skipping to check-out)"
                    self.logger.warning(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                if context.is_active() and self._server_cfg['hp_io_init'] and not self._is_hp_io_valid():
                    emsg = (f"The hp_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['hp_io_init'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)
                # activate the writer
                self._rw_lock_state['aw'] += 1
                active = True
                self._active_clients[tid] = urllib.parse.unquote(context.peer())
                self.logger.debug(f"(writer) check-in (end):\t\t{self._rw_lock_state=}")
                # END check-in critical section
            yield None
        except RuntimeError as err:
            pass
        finally:
            with self._hp_io_lock:
                # BEGIN check-out critical section
                self.logger.debug(f"(writer) check-out (start):\t{self._rw_lock_state=}")
                if active:  # handle edge cases where thread is interrupted or has an error during lock acquire
                    self._rw_lock_state['aw'] = self._rw_lock_state['aw'] - 1  # no longer active
                    del self._active_clients[tid]
                # Wake up waiting readers or a waiting writer (prioritize waiting writers).
                if self._rw_lock_state['ww'] > 0:  # Give lock priority to waiting writers
                    self._write_ok_condvar.notify()
                elif self._rw_lock_state['wr'] > 0:
                    self._read_ok_condvar.notify_all()
                self.logger.debug(f"(writer) check-out (end):\t{self._rw_lock_state=}")
                # END check-out critical section

    @contextmanager
    def _rw_lock_reader(self, context):
        read_fmap_idx = -1  # remember which read_queue freemap entry corresponds to this thread
        tid = threading.get_ident()
        active = False
        try:
            with self._hp_io_lock:
                # BEGIN check-in critical section
                self.logger.debug(f"(reader) check-in (start):\t{self._rw_lock_state=}"
                                  f"\n{self._read_queues_freemap=}")
                # Wait until no active writers or waiting writers
                while context.is_active() and (self._rw_lock_state['aw'] + self._rw_lock_state['ww']) > 0:
                    self._rw_lock_state['wr'] += 1
                    self._read_ok_condvar.wait()
                    self._rw_lock_state['wr'] -= 1

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

                # check if hp_io is valid
                if context.is_active() and self._server_cfg['hp_io_init'] and not self._is_hp_io_valid():
                    emsg = (f"the hp_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['hp_io_init'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # activate the reader
                self._rw_lock_state['ar'] += 1
                active = True
                self._active_clients[tid] = urllib.parse.unquote(context.peer())
                self.logger.debug(f"(reader) check-in (end):\t\t{self._rw_lock_state=}, fmap_idx={read_fmap_idx}"
                                  f"\n{self._read_queues_freemap=}")
                # END check-in critical section
            yield read_fmap_idx
        finally:
            with self._hp_io_lock:
                # BEGIN check-out critical section
                self.logger.debug(f"(reader) check-out (start):\t{self._rw_lock_state=}")
                if active:
                    self._rw_lock_state['ar'] = self._rw_lock_state['ar'] - 1  # no longer active
                    del self._active_clients[tid]
                    self._read_queues_freemap[read_fmap_idx] = False  # release the read queue
                # Wake up waiting readers or a waiting writer (prioritize waiting writers).
                if self._rw_lock_state['ar'] == 0 and self._rw_lock_state['ww'] > 0:
                    self._write_ok_condvar.notify()
                elif self._rw_lock_state['wr'] > 0:
                    self._read_ok_condvar.notify_all()
                self.logger.debug(f"(reader) check-out (end):\t\t{self._rw_lock_state=}")
                # END check-out critical section

    def _start_hp_io_thread(self, hp_io_cfg):
        """Creates a new hp_io thread with the given f9t_cfg.
        @return: True iff the hp_io thread was created and established a valid connection to the target F9t chip
        """
        # Terminate any currently alive hp_io thread
        self._stop_hp_io_thread()  # no effect if a hp_io thread is not alive

        # Create new hp_io_thread using the client's configuration
        self._stop_io.clear()
        self._hp_io_thread = Thread(
            target=hp_io_data_DEBUG,
            args=(
                hp_io_cfg["named_pipe_path"],
                hp_io_cfg["timeout"],
                self._read_queues,
                self._read_queues_freemap,
                self._stop_io,
                self._hp_io_valid,
                self.logger,
            ),
            kwargs={
                "early_exit": False,  # causes the hp_io thread have a fatal exception after the given delay
                "early_exit_delay_seconds": 25
            },
            daemon=False,
        )
        self._hp_io_thread.start()

        # check if thread could be properly initialized
        self._hp_io_valid.wait(1)
        if self._is_hp_io_valid():
            self.logger.info("hp_io thread alive and valid")
            return True
        else:
            self._stop_hp_io_thread()
            return False

    def _is_hp_io_valid(self):
        if self._hp_io_thread is not None and self._hp_io_thread.is_alive() and self._hp_io_valid.is_set():
            return True
        elif self._hp_io_thread is None:
            self.logger.warning("hp_io thread is uninitialized")
        elif not self._hp_io_thread.is_alive():
            self.logger.critical("hp_io thread is not alive")
        elif not self._hp_io_valid.is_set():
            self.logger.warning("hp_io thread is alive but not valid")
        else:
            emsg = (f"unhandled is_hp_io_valid case: "
                    f"{self._hp_io_thread=}, "
                    f"{self._hp_io_thread.is_alive()=},"
                    f"{self._hp_io_valid=}")
            self.logger.critical(emsg)
            raise RuntimeError(emsg)  # SHOULD NEVER REACH HERE
        return False

    def _stop_hp_io_thread(self):
        """Stops the hp_io thread. Idempotent behavior.
        @return: True iff the hp_io thread is not alive."""
        self._stop_io.set()  # signal hp_io thread to exit gracefully
        if self._hp_io_thread is not None and self._hp_io_thread.is_alive():
            try:
                self._hp_io_thread.join(5)  # wait until hp_io exits
            except RuntimeError as rerr:
                self.logger.critical(f"encountered runtime error while stopping hp_io thread: {rerr}")
                # raise rerr
            finally:
                if self._hp_io_thread.is_alive():  # check if join succeeded or timeout happened while waiting
                    self.logger.critical(f"Could not stop_io hp_io thread")
                    return False
                else:
                    self.logger.info(f"Successfully terminated hp_io thread")
                    return True
        else:
            self.logger.debug("no hp_io thread to stop_io (doing nothing)")
            return True

    def CaptureScience(self, request, context):
        """Forward u-blox packets to the client. [reader]"""
        # unpack the requested message pattern filters
        self.logger.info(f"new CaptureScience rpc from {urllib.parse.unquote(context.peer())}")
        # TODO: check if the patterns are valid
        with (self._rw_lock_reader(context) as rid):  # rid = allocated reader id
            # BEGIN critical section for F9t [read] access
            # Clear the read_queue of old data
            rq = self._read_queues[rid]
            while not rq.empty():
                rq.get()
            if self._server_cfg['hp_io_init'] and self._is_hp_io_valid():
                # self.logger.info("Streaming messages")
                while context.is_active() and self._is_hp_io_valid():
                    # self.logger.debug("waiting for input")
                    try:
                        # wait for next packet from the hp_io thread
                        # add a timeout of to avoid starvation in case the hp_io thread unexpectedly exits while this thread is blocking on the read_queue
                        parsed_data = rq.get(timeout=10)

                        send_timestamp = timestamp_pb2.Timestamp()
                        send_timestamp.GetCurrentTime()

                        # TODO: get these values from data_config.json
                        image_shape = [32, 32]
                        bytes_per_pixel = 2

                        pano_image = PanoImage(
                            type=parsed_data["type"],
                            header=ParseDict(parsed_data["header"], Struct()),
                            image_array=parsed_data["image_array"].flatten().tolist(),
                            image_shape=image_shape,
                            bytes_per_pixel=bytes_per_pixel
                        )

                        capture_science_response = CaptureScienceResponse(
                            type=CaptureScienceResponse.Type.DATA,
                            name="test_movie_data",
                            timestamp=send_timestamp,
                            message="testing",
                            pano_image=pano_image
                        )
                        yield capture_science_response
                    except queue.Empty:
                        self.logger.warning("hp_io thread may have stopped sending data")
                        continue

                # log reason why streaming stopped
                if not context.is_active():
                    self.logger.info(f"CaptureScience client disconnected")
                else:
                    emsg = (f"The hp_io thread data stream unexpectedly became invalid! "
                            f"Check the server logs to debug this issue")
                    self.logger.critical(emsg)
                    context.abort(grpc.StatusCode.INTERNAL, emsg)
            else:
                # TODO: implement InitHpIO
                emsg = "Uninitialized hp_io thread. Run InitHpIo with a valid hp_io configuration to initialize it."
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            # END critical section for hp_io [read] access

def serve(server_cfg):
    """Create the gRPC server threadpool and start providing the UbloxControl service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_cfg['max_workers']))
    daq_data_pb2_grpc.add_DaqDataServicer_to_server(
        DaqDataServicer(server_cfg), server
    )

    # Add RPC reflection to show available commands to users
    SERVICE_NAMES = (
        daq_data_pb2.DESCRIPTOR.services_by_name["DaqData"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start gRPC and configure to listen on port 50051
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"The gRPC services {SERVICE_NAMES} are running.\nEnter CTRL+C to stop_io them.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        grace = server_cfg["shutdown_grace_period"]
        print(f"'^C' received, shutting down the server in {grace} seconds.")
        server.stop(grace=grace).wait(grace)
        sys.exit(0)



if __name__ == "__main__":
    # Load server configuration
    server_cfg_file = "daq_data_server_config.json"
    with open(cfg_dir / server_cfg_file, "r") as f:
        server_cfg = json.load(f)
    serve(server_cfg)
