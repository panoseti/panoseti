#!/usr/bin/env python3

"""
The Python implementation of a gRPC DaqData server.

Requires following to function correctly:
    1. A POSIX-compliant operating system.
    2. All Python packages specified in requirements.txt.
    3. A connection to a panoseti module.
"""
from concurrent import futures
from threading import Event, Thread
from queue import Queue
from glob import glob
from contextlib import contextmanager
import logging
import queue
import json
import sys
import threading
import time
import urllib.parse

## --- gRPC imports ---
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
from daq_data_pb2 import PanoImage, TestCase, StreamImagesResponse, StreamImagesRequest

## --- daq_data utils ---
from daq_data_resources import make_rich_logger
from daq_data_testing import *

## --- panoseti utils ---
sys.path.append("../../util")
import pff, config_file
sys.path.append("../../control")
import util


""" hp_io test macros """
PH_PFF = "start_2024-07-25T04_34_46Z.dp_ph256.bpp_2.module_1.seqno_0.debug_TRUNCATED.pff"
IMG_PFF = "start_2024-07-25T04_34_46Z.dp_img16.bpp_2.module_1.seqno_0.debug_TRUNCATED.pff"
MOVIE_TYPE = 'img16'
PH_TYPE = 'ph256'

SIM_DATA_DIR = Path("test_env")

SIM_RUN_DIR = SIM_DATA_DIR / Path("module_1/obs_SIMULATE")
os.makedirs(SIM_RUN_DIR, exist_ok=True)
DAQ_ACTIVE_FILE = SIM_RUN_DIR / "daq_active"
MOVIE_DST   = SIM_RUN_DIR / IMG_PFF
PH_DST      = SIM_RUN_DIR / PH_PFF

REAL_RUN_DIR = SIM_DATA_DIR / Path("obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd")
MOVIE_SRC   = REAL_RUN_DIR / IMG_PFF
PH_SRC      = REAL_RUN_DIR / PH_PFF

def hp_sim_thread_fn(
    dp_cfg: Dict[str, Any],
    update_interval: float,
    stop_io: Event,
    logger: logging.Logger
) -> None:
    """Simulate hashpipe data stream: Read a real file and write to a fake file. """
    logger.info("hp_sim thread started")
    # prevent multiple server instances from running this thread
    if os.path.exists(DAQ_ACTIVE_FILE):
        logger.critical("hp_sim thread exited: another server instance is already running!")
        sys.exit()

    with open(DAQ_ACTIVE_FILE, "w") as daq_active:
        daq_active.write("1")
    try:
        with open(MOVIE_DST, "wb") as movie_dst, \
        open(MOVIE_SRC, "rb") as movie_src, \
        open(PH_DST, "wb") as ph_dst, \
        open(PH_SRC, "rb") as ph_src:
            while not stop_io.is_set():
                # get file info, e.g. frame size from the ph and img source files
                (movie_frame_size, movie_nframes, first_t, last_t) = pff.img_info(movie_src, dp_cfg[MOVIE_TYPE]['bytes_per_image'])
                movie_src.seek(0, os.SEEK_SET)
                logger.info(f"movie src: {movie_frame_size=}, {movie_nframes=}")

                (ph_frame_size, ph_nframes, first_t, last_t) = pff.img_info(ph_src, dp_cfg[PH_TYPE]['bytes_per_image'])
                logger.info(f"ph src: {ph_frame_size=}, {ph_nframes=}, {first_t=}, {last_t=}")
                ph_src.seek(0, os.SEEK_SET)

                # copy frames from fsrc to fdst to simulate data acquisition software
                ph_i = movie_i = 0
                while not stop_io.is_set() and ph_i < ph_nframes and movie_i < movie_nframes:
                    ph_data = ph_src.read(ph_frame_size)
                    ph_nbytes_written = ph_dst.write(ph_data)

                    movie_data = movie_src.read(movie_frame_size)
                    movie_nbytes_written = movie_dst.write(movie_data)

                    ph_dst.flush()
                    movie_dst.flush()

                    ph_i += 1
                    movie_i += 1

                    # logger.info(f"{ph_nbytes_written=}, {movie_nbytes_written=}")
                    time.sleep(update_interval)
    finally:
        os.unlink(DAQ_ACTIVE_FILE)
        os.unlink(MOVIE_DST)
        os.unlink(PH_DST)
        logger.info("hp_sim thread exited")


def hp_io_thread_fn(
        dp_cfg: Dict[str, Any],
        update_interval: float,
        reader_states: List[Dict],
        stop_io: Event,
        valid: Event,
        logger: logging.Logger,
        **kwargs
) -> None:
    """ Receive pulse-height and movie-mode data from hashpipe and broadcast it to all active reader queues. """
    logger.info(f"Created a new hp_io thread with the following options: {kwargs=}")
    valid.clear()  # indicate hashpipe io channel is currently invalid
    # parse any kwargs
    if "early_exit_delay_seconds" in kwargs:
        early_exit_counter = kwargs["early_exit_delay_seconds"]
    else:
        early_exit_counter = 30
    try:
        data_dir = SIM_DATA_DIR
        module_id = 1

        # wait until there is an in-progress run
        while not stop_io.is_set():
            run_pattern = f"{data_dir}/module_{module_id}/obs_*"
            runs = glob(run_pattern)
            nruns = len(runs)
            if nruns == 0:
                raise FileNotFoundError(f'no run of module {module_id} in {run_pattern}')
            run_path = sorted(runs)[-1]
            # TODO: check if this run is in progress (with production code)
            if os.path.exists(DAQ_ACTIVE_FILE):
                break
            logger.info("Waiting for in-progress run to start")
            time.sleep(1)
            # run = util.daq_get_run_name()
            # if not run:
            #     logger.error('no run')
            #     return

        def init_dp_cfg():
            """initialize dp_cfg with run-specific information and wait until hashpipe starts"""
            for dp in dp_cfg:
                # Get the current number of pff files with type [dp]
                dp_cfg[dp]['glob_pat'] = '%s/*%s*.pff' % (run_path, dp)
                # wait until hashpipe starts writing files
                nfiles = 0
                while not stop_io.is_set() and nfiles == 0:
                    files = glob(dp_cfg[dp]['glob_pat'])
                    nfiles = len(files)
                    logger.debug(f'no file of type {dp} in {dp_cfg[dp]["glob_pat"]}')
                    time.sleep(0.5)
                file = sorted(files)[-1]

                # wait until the filesize is large enough to read one image of type [dp]
                filepath = file
                while not stop_io.is_set():
                    if os.path.getsize(filepath) >= dp_cfg[dp]['bytes_per_image']:
                        break
                    time.sleep(0.5)

                # read the first frame of the file to determine the frame_size (this size is constant for the entire run)
                f = open(filepath, 'rb')
                logger.debug(f"{dp=}: {filepath=}: {dp_cfg[dp]['bytes_per_image']=}")
                (frame_size, nframes, first_t, last_t) = pff.img_info(f, dp_cfg[dp]['bytes_per_image'])
                dp_cfg[dp]['frame_size'] = frame_size

                f.seek(0, os.SEEK_SET)
                dp_cfg[dp]['f'] = f
                dp_cfg[dp]['nfiles'] = nfiles
                dp_cfg[dp]['filepath'] = filepath
                dp_cfg[dp]['last_frame'] = -1

        def dp_main(d: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[int, ...]] or Tuple[None, None]:
            """
            Check if there is new pff data of type [dp].
            If new data is present:
                1. Update dp_cfg[dp] accordingly.
                2. return a tuple of (pff header, pff image).
            Otherwise, return (None, None)
            Note: this function mutates dp_cfg.
            """
            f = d['f']
            nfiles = d['nfiles']
            filepath = d['filepath']
            last_frame = d['last_frame']
            try:
                files = glob(d['glob_pat'])
                if len(files) > nfiles:
                    nfiles = len(files)
                    f.close()
                    file = sorted(files)[-1]
                    filepath = file
                    f = open(filepath, 'rb')
                    last_frame = -1
                fsize = f.seek(0, os.SEEK_END)
                nframes = int(fsize / d['frame_size'])
                # check if any new frames have been written to this file
                if nframes > last_frame + 1:
                    # seek to the latest frame in the file
                    last_frame = nframes - 1
                    f.seek(last_frame * d['frame_size'], os.SEEK_SET)

                    # parse pff header and image
                    try:
                        header_str = pff.read_json(f)
                        img = pff.read_image(f, d['image_shape'][0], d['bytes_per_pixel'])
                        # the check below is necessary to handle the rare case where a pff file has
                        # reached the max size specified in data_config.json resulting in no data for the last frame.
                        if header_str and img:
                            header = json.loads(header_str)
                            return header, img
                    except Exception as e:
                        logger.error(f"Failed to read pff header and image from file {filepath} with error: {e}")
                        return None, None
                return None, None
            finally:
                # always update dp_cfg upon exit.
                # important for ensuring we always close any newly opened file pointers
                d['f'] = f
                d['nfiles'] = nfiles
                d['filepath'] = filepath
                d['last_frame'] = last_frame

        # signal the hp_io thread is ready to service client requests for data preview
        init_dp_cfg()
        valid.set()
        while not stop_io.is_set():
            for dp in dp_cfg:
                d = dp_cfg[dp]
                header, img = dp_main(d)
                if header and img:
                    # create PanoImage message from the latest image
                    pano_image = PanoImage(
                        type=d['pano_image_type'],
                        header= ParseDict(header, Struct()),
                        image_array=img,
                        shape=d['image_shape'],
                        bytes_per_pixel=d['bytes_per_pixel'],
                        file=os.path.basename(d['filepath']),
                        frame_number=d['last_frame'],
                    )
                    # create object to pass to each waiting writer
                    parsed_data = {
                        "pano_image": pano_image,
                    }

                    # broadcast image data to all waiting clients
                    for rs in [rs for rs in reader_states if rs['is_allocated']]:
                        rq = rs['queue']
                        if d['is_ph'] and rs['config']['stream_pulse_height_data']:
                            rq.put(parsed_data)
                        elif not d['is_ph'] and rs['config']['stream_movie_data']:
                            rq.put(parsed_data)

                if "early_exit" in kwargs and kwargs["early_exit"]:
                    early_exit_counter -= 1
                    if early_exit_counter == 0:
                        raise TimeoutError("test hp_io thread unexpected termination")
                time.sleep(update_interval)
    except Exception as err:
        logger.critical(f"hp_io thread encountered a fatal exception! {err}")
        raise err
    finally:
        # close any open file pointers
        for dp in dp_cfg:
            if 'f' in dp_cfg[dp]:
                dp_cfg[dp]['f'].close()
        valid.clear()
        logger.info("hp_io thread exited")


"""gRPC server implementing DaqData RPCs"""
class DaqDataServicer(daq_data_pb2_grpc.DaqDataServicer):
    """Provides implementations for DaqData RPCs."""

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
        self._active_clients = {}  # dict of tid : {"client_ip":context.peer(), "thread": Thread} for debugging

        self._server_cfg = server_cfg

        # Create the server's logger
        self.logger = make_rich_logger(__name__, level=logging.DEBUG)

        # Load default hahspipe_io configuration
        with open(cfg_dir/self._server_cfg["default_hp_io_config_file"], "r") as f:
            self._hp_io_cfg = json.load(f)

        # State for single producer, multiple consumer hp_io access
        # A single IO thread manages the dataflow between multiple concurrent RPC threads and the hp_io thread:
        #   [single RPC writer -> hp_io thread] send messages to the hp_io thread
        #   [hp_io thread -> many RPC readers] broadcast image data to active read_queues
        self._hp_io_thread: Thread = None

        # Initialize an array of reader_state dicts to support up to max_worker concurrent reader RPCs
        self._reader_states: List[Dict[str, Any]] = []
        # _reader_states is a list of reader gRPC state dictionaries
        #   - "is_allocated": True iff corresponding queue is allocated to a reader
        #   - "queue": Queue implementing single producer (hp_io), multiple independent consumer model
        #   - "config": Keyword configuration options
        for _ in range(server_cfg['max_workers']):
            default_config = {
                "stream_movie_data": True,
                "stream_pulse_height_data": True,
                "stream_hashpipe_status": False,
                "update_interval_seconds": 1,
            }
            default_reader_state = {
                "is_allocated": False,
                "queue": Queue(maxsize=server_cfg['max_read_queue_size']),
                "config": default_config,
            }
            self._reader_states.append(default_reader_state)
        self._stop_io = Event()  # Signals hp_io thread to exit
        self._hp_io_valid = Event()  # Set only if the hp_io thread is active and collecting data
        self._shutdown_event = Event()  # Set only at shutdown

        # Start the hp_io thread if server_cfg points to a valid hp_io_cfg
        if self._server_cfg["allow_init_from_default"] and self._hp_io_cfg["valid_config"]:
            self.logger.info(f"Creating the initial hp_io thread from config: "
                             f"{self._server_cfg['allow_init_from_default']=} and "
                             f"{self._hp_io_cfg['valid_config']=}.")
            self._server_cfg['hp_io_init'] = True
            self._start_hp_io_thread(self._hp_io_cfg)
        else:
            self.logger.warning(f"An InitHpIo call is required to start the hp_io thread: "
                                f"{self._server_cfg['allow_init_from_default']=} and "
                                f"{self._hp_io_cfg['valid_config']=}.")

            self._server_cfg['hp_io_init'] = False

    def shutdown(self):
        shutdown_record = {}
        self._server_cfg['hp_io_init'] = False
        # signal hp_io thread to exit gracefully
        self._stop_io.set()
        # signal any blocking readers to wake up and exit
        self._shutdown_event.set()
        with self._hp_io_lock:
            self._read_ok_condvar.notify_all()
            for rs in [rs for rs in self._reader_states if rs['is_allocated']]:
                try:
                    rs['queue'].put_nowait("shutdown")
                except queue.Full:
                    pass
        # wait for the hp_io thread to exit
        shutdown_record['stop_hp_io'] = self._stop_hp_io_thread(2)
        # wait for active server threads to exit
        for ac in self._active_clients.values():
            ac['thread'].join()
        active_clients = [ac["client_ip"] for ac in self._active_clients.values()]
        if len(active_clients) > 0:
            self.logger.warning(f"active clients at shutdown: {active_clients=}")
        shutdown_record['stop_active_clients'] = len(active_clients) == 0

        # check if state was updated properly
        lock_status_ok = True
        for thread_state, num_threads in self._rw_lock_state.items():
            if num_threads != 0:
                self.logger.critical(f"[rw lock] unexpected threads in state {thread_state} at termination!\n"
                                     f"{self._rw_lock_state=}")
        shutdown_record['lock_status_ok'] = lock_status_ok
        if all(shutdown_record.values()):
            self.logger.info("Successfully released all resources")
        else:
            self.logger.critical(f"Some server resources were not released: {shutdown_record=}")


    @contextmanager
    def _rw_lock_writer(self, context):
        tid = threading.get_ident()
        active = False
        try:
            with self._hp_io_lock:
                # BEGIN check-in critical section
                # All reader RPCs are long-lived server streaming operations.
                # The server's synchronization logic will prevent updates to _server_cfg while any reader RPCs are active,
                # so we should cancel any writer RPCs immediately

                if self._rw_lock_state['ar'] > 0:
                    active_clients = str([c["client_ip"] for c in self._active_clients.values()])
                    emsg = (f"Cannot modify server state because there are {self._rw_lock_state['ar']} active "
                            f"StreamImages clients. Stop these client processes then try again: {active_clients=}.")
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
                self.logger.debug(f"(writer) check-in (start):\t{self._rw_lock_state=}")

                # Wait until no active readers or active writers
                while (not self._shutdown_event.is_set() and
                       context.is_active() and
                       (self._rw_lock_state['aw'] + self._rw_lock_state['ar']) > 0):
                    self._rw_lock_state['ww'] += 1
                    self._write_ok_condvar.wait(timeout=5)
                    self._rw_lock_state['ww'] -= 1

                # check if the server is still active
                if self._shutdown_event.is_set():
                    emsg = "server shutdown initiated during writer lock acquisition [skipping to check-out]"
                    self.logger.error(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                # check if the client is still active
                if not context.is_active():
                    emsg = "client cancelled rpc during writer lock acquisition (skipping to check-out)"
                    self.logger.warning(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                # check if the hp_io thread is valid
                if self._server_cfg['hp_io_init'] and not self._is_hp_io_valid():
                    emsg = (f"The hp_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['hp_io_init'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # activate the writer
                self._rw_lock_state['aw'] += 1
                active = True
                self._active_clients[tid] = {
                    "client_ip": urllib.parse.unquote(context.peer()),
                    "thread": threading.current_thread(),
                    "type": "writer",
                }
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
        reader_idx = -1  # remember which reader_states dict corresponds to this thread
        tid = threading.get_ident()
        active = False
        try:
            with self._hp_io_lock:
                # BEGIN check-in critical section
                self.logger.debug(f"(reader) check-in (start):\t{self._rw_lock_state=}"
                                  f"\n{[rs['is_allocated'] for rs in self._reader_states]=}")
                # Wait until no active writers or waiting writers
                while (not self._shutdown_event.is_set() and
                       context.is_active() and
                       (self._rw_lock_state['aw'] + self._rw_lock_state['ww']) > 0):
                    self._rw_lock_state['wr'] += 1
                    self._read_ok_condvar.wait()
                    self._rw_lock_state['wr'] -= 1

                # check if the server is still active
                if self._shutdown_event.is_set():
                    emsg = "server shutdown initiated during reader lock acquisition [skipping to check-out]"
                    self.logger.error(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                # check if the client is still active
                if not context.is_active():
                    emsg = "client context terminated during reader lock acquisition [skipping to check-out]"
                    self.logger.error(emsg)
                    context.cancel()

                # check if the hp_io thread is valid
                if self._server_cfg['hp_io_init'] and not self._is_hp_io_valid():
                    emsg = (f"The hp_io thread data stream is unexpectedly invalid!"
                            f" (skipping to check-out)")
                    self.logger.critical(emsg)
                    self._server_cfg['hp_io_init'] = False
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # allocate reader resources
                for idx, rs in enumerate(self._reader_states):
                    if not rs['is_allocated']:
                        reader_idx = idx
                        self._reader_states[idx]['is_allocated'] = True
                        break

                # check if the allocation succeeded
                if reader_idx < 0:
                    emsg = "reader_states allocation failed during reader check-in! [SHOULD NEVER HAPPEN]"
                    self.logger.critical(emsg)
                    context.abort(grpc.StatusCode.INTERNAL, emsg)

                # activate the reader
                self._rw_lock_state['ar'] += 1
                active = True
                self._active_clients[tid] = {
                    "client_ip": urllib.parse.unquote(context.peer()),
                    "thread": threading.current_thread(),
                    "type": "reader",
                }
                self.logger.debug(f"(reader) check-in (end):\t\t{self._rw_lock_state=}, fmap_idx={reader_idx}"
                                  f"\n{[rs['is_allocated'] for rs in self._reader_states]=}")
                # END check-in critical section
            yield self._reader_states[reader_idx]
        finally:
            with self._hp_io_lock:
                # BEGIN check-out critical section
                self.logger.debug(f"(reader) check-out (start):\t{self._rw_lock_state=}")
                if active:
                    self._rw_lock_state['ar'] = self._rw_lock_state['ar'] - 1  # no longer active
                    del self._active_clients[tid]
                    self._reader_states[idx]['is_allocated'] = False # release reader resources
                # Wake up waiting readers or a waiting writer (prioritize waiting writers).
                if self._rw_lock_state['ar'] == 0 and self._rw_lock_state['ww'] > 0:
                    self._write_ok_condvar.notify()
                elif self._rw_lock_state['wr'] > 0:
                    self._read_ok_condvar.notify_all()
                self.logger.debug(f"(reader) check-out (end):\t\t{self._rw_lock_state=}")
                # END check-out critical section

    def get_dp_cfg(self, dps):
        """Returns a dictionary of static properties for the given data products."""
        dp_cfg = {}
        for dp in dps:
            if dp == 'img16' or dp == 'ph1024':
                image_shape = [32, 32]
                bytes_per_pixel = 2
            elif dp == 'img8':
                image_shape = [32, 32]
                bytes_per_pixel = 1
            elif dp == 'ph256':
                image_shape = [16, 16]
                bytes_per_pixel = 2
            else:
                raise Exception("bad data product %s" % dp)
            bytes_per_image = bytes_per_pixel * image_shape[0] * image_shape[1]
            is_ph = 'ph' in dp
            # Get type enum for PanoImage message
            if is_ph:
                pano_image_type = PanoImage.Type.PULSE_HEIGHT
            else:
                pano_image_type = PanoImage.Type.MOVIE

            dp_cfg[dp] = {
                "image_shape": image_shape,
                "bytes_per_pixel": bytes_per_pixel,
                "bytes_per_image": bytes_per_image,
                "is_ph": is_ph,
                "pano_image_type": pano_image_type,
            }
        return dp_cfg

    def _start_hp_io_thread(self, hp_io_cfg):
        """Creates a new hp_io thread with the given hp_io_cfg.
        @return: True iff the hp_io thread was created and attached to a valid active observing run.
        """
        # Terminate any currently alive hp_io thread
        self._stop_hp_io_thread(5)  # no effect if a hp_io thread is not alive

        dps = ["img16", "ph256"]
        dp_cfg = self.get_dp_cfg(dps)

        # Create a new hp_io_thread using the client's configuration
        self._stop_io.clear()
        self._hp_io_thread = Thread(
            target=hp_io_thread_fn,
            args=(
                dp_cfg.copy(),
                max(self._hp_io_cfg['update_interval_seconds'], 0.25),
                self._reader_states,
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
        hp_sim_thread = Thread(
            target=hp_sim_thread_fn,
            args=(
                dp_cfg.copy(),
                max(self._hp_io_cfg['update_interval_seconds'] / 2, 0.1),
                self._stop_io,
                self.logger
            )
        )
        self._hp_io_thread.start()
        hp_sim_thread.start()

        # check if thread could be properly initialized
        self._hp_io_valid.wait(5)
        if self._is_hp_io_valid():
            self.logger.info("hp_io thread alive and valid")
            return True
        else:
            self._stop_hp_io_thread(5)
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

    def _stop_hp_io_thread(self, timeout:float=5.0):
        """Stops the hp_io thread. Idempotent behavior.
        @return: True iff the hp_io thread is not alive.
        :param timeout: seconds to wait for hp_io thread to exit gracefully"""
        self._stop_io.set()  # signal hp_io thread to exit gracefully
        if self._hp_io_thread is not None and self._hp_io_thread.is_alive():
            try:
                self._hp_io_thread.join(timeout)  # wait until hp_io exits
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

    def StreamImages(self, request, context):
        """Forward sample panoseti movie and pulse-height images to the client. [reader]"""
        # unpack the requested message pattern filters
        self.logger.info(f"new StreamImages rpc from {urllib.parse.unquote(context.peer())}")
        with self._rw_lock_reader(context) as reader_state:  # rid = allocated reader id for indexing into shared reader resources
            # BEGIN reader critical section
            # Clear old data from the read_queue
            rq = reader_state['queue']
            while not rq.empty():
                rq.get()
            # Set stream filter options
            reader_state['config']['stream_movie_data'] = request.stream_movie_data
            reader_state['config']['stream_pulse_height_data'] = request.stream_pulse_height_data
            reader_state['config']['update_interval_seconds'] = request.update_interval_seconds
            # Validate client request
            if not (self._server_cfg["min_client_update_interval_seconds"]
                    <= request.update_interval_seconds
                    <= self._server_cfg['max_client_update_interval_seconds']):
                emsg = (f"update_interval_seconds must be in the interval "
                        f"[{self._server_cfg['min_client_update_interval_seconds']}, "
                        f"{self._server_cfg['max_client_update_interval_seconds']}"
                        f"seconds. Got {request.update_interval_seconds}")
                self.logger.critical(emsg)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            elif not request.stream_movie_data and not request.stream_pulse_height_data:
                emsg = "At least one of the stream flags must be set to True"
                self.logger.info(emsg)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            elif not self._server_cfg['hp_io_init']:
                # TODO: implement InitHpIO
                emsg = "Uninitialized hp_io thread. Run InitHpIo with a valid hp_io configuration to initialize it."
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            # Valid server state -> start streaming!
            while context.is_active() and self._is_hp_io_valid() and not self._shutdown_event.is_set():
                try:
                    # wait for next packet from the hp_io thread
                    # add a timeout to avoid starvation if hp_io thread unexpectedly exits while this thread is blocking on the read_queue
                    parsed_data = rq.get(timeout=request.update_interval_seconds)
                    if self._shutdown_event.is_set():
                        emsg = "server shutdown initiated"
                        context.abort(grpc.StatusCode.CANCELLED, emsg)

                    send_timestamp = timestamp_pb2.Timestamp()
                    send_timestamp.GetCurrentTime()

                    # TODO: get these values from data_config.json
                    pano_image = parsed_data['pano_image']
                    pano_type = PanoImage.Type.Name(pano_image.type)

                    stream_images_response = StreamImagesResponse(
                        name=f"StreamImageResponse [Data]",
                        timestamp=send_timestamp,
                        message=f"",
                        pano_image=pano_image
                    )

                    yield stream_images_response
                except queue.Empty:
                    # self.logger.debug("hp_io thread may have stopped sending data")
                    continue

            # log reason why streaming stopped
            if not context.is_active():
                self.logger.info(f"StreamImages client disconnected")
            elif not self._stop_io.is_set():
                emsg = (f"The hp_io thread data stream unexpectedly became invalid! "
                        f"Check the server logs to debug this issue")
                self.logger.critical(emsg)
                context.abort(grpc.StatusCode.INTERNAL, emsg)
            # END reader critical section


def serve(server_cfg):
    """Create the gRPC server threadpool and start providing the UbloxControl service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_cfg['max_workers']))
    daq_data_servicer = DaqDataServicer(server_cfg)
    daq_data_pb2_grpc.add_DaqDataServicer_to_server(
        daq_data_servicer, server
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
        daq_data_servicer.shutdown()
        server.stop(grace=grace).wait(grace)
        sys.exit(0)



if __name__ == "__main__":
    # Load server configuration
    cfg_dir = Path('config')
    default_hp_io_thread_config_file = 'default_hp_io_config.json'

    # Configuration
    with open(cfg_dir / default_hp_io_thread_config_file) as f:
        default_hp_io_thread_config = json.load(f)

    server_cfg_file = "daq_data_server_config.json"
    with open(cfg_dir / server_cfg_file, "r") as f:
        server_cfg = json.load(f)
    serve(server_cfg)
