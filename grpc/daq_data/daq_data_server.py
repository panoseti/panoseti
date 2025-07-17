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
from daq_data_pb2 import PanoImage, TestCase, StreamImagesResponse, StreamImagesRequest, InitHpIoResponse

## --- daq_data utils ---
from daq_data_resources import make_rich_logger
from daq_data_testing import *

## --- panoseti utils ---
sys.path.append("../../util")
import pff, config_file
sys.path.append("../../control")
import util


""" hp_io test macros """
PH_PFF = "start_2024-07-25T04_34_46Z.dp_ph256.bpp_2.module_1.seqno_{seqno}.debug_TRUNCATED.pff"
IMG_PFF = "start_2024-07-25T04_34_46Z.dp_img16.bpp_2.module_1.seqno_{seqno}.debug_TRUNCATED.pff"
MOVIE_TYPE = 'img16'
PH_TYPE = 'ph256'

SIM_DATA_DIR = Path("test_env")

SIM_RUN_DIR = SIM_DATA_DIR / Path("module_1/obs_SIMULATE")
os.makedirs(SIM_RUN_DIR, exist_ok=True)
DAQ_ACTIVE_FILE = SIM_RUN_DIR / "daq_active"
def get_sim_movie_dest(seqno):
    return SIM_RUN_DIR / IMG_PFF.format(seqno=seqno)
def get_sim_ph_dest(seqno):
    return SIM_RUN_DIR / PH_PFF.format(seqno=seqno)

REAL_RUN_DIR = SIM_DATA_DIR / Path("obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd")
MOVIE_SRC   = REAL_RUN_DIR / IMG_PFF.format(seqno=0)
PH_SRC      = REAL_RUN_DIR / PH_PFF.format(seqno=0)


def is_daq_active(simulate_daq=False):
    """Returns True iff the data stream from hashpipe or simulated hashpipe is active."""
    daq_active = False
    if simulate_daq:
        daq_active = os.path.exists(DAQ_ACTIVE_FILE)
    else:
        daq_active = util.is_hashpipe_running()
    return daq_active




def daq_sim_thread_fn(
    dp_cfg: Dict[str, Any],
    update_interval: float,
    stop_io: Event,
    logger: logging.Logger,
    frames_per_pff = 20,
    **kwargs,
) -> None:
    """Simulate hashpipe data stream: Read a real file and write to a fake file. """
    if "early_exit_delay_seconds" in kwargs:
        early_exit_counter = kwargs["early_exit_delay_seconds"]
    else:
        early_exit_counter = 30
    logger.info("hp_sim thread started")
    # prevent multiple server instances from running this thread
    if os.path.exists(DAQ_ACTIVE_FILE):
        emsg = "hp_sim thread is already running on another server instance!"
        logger.critical(emsg)
        raise RuntimeError(emsg)

    simulated_data_files = []
    try:
        with open(DAQ_ACTIVE_FILE, "w") as daq_active:
            daq_active.write("1")

        with open(MOVIE_SRC, "rb") as movie_src, open(PH_SRC, "rb") as ph_src:
            # get file info, e.g. frame size from the ph and img source files
            (movie_frame_size, movie_nframes, first_t, last_t) = pff.img_info(movie_src, dp_cfg[MOVIE_TYPE]['bytes_per_image'])
            movie_src.seek(0, os.SEEK_SET)
            logger.info(f"movie src: {movie_frame_size=}, {movie_nframes=}")

            (ph_frame_size, ph_nframes, first_t, last_t) = pff.img_info(ph_src, dp_cfg[PH_TYPE]['bytes_per_image'])
            logger.info(f"ph src: {ph_frame_size=}, {ph_nframes=}")
            ph_src.seek(0, os.SEEK_SET)
            # copy frames from [dp]_src to dp_dst to simulate data acquisition software
            fnum = 0
            seqno = 0
            while not stop_io.is_set() and fnum < min(ph_nframes, movie_nframes):
                # Every [frames_per_pff] frames, create a new file of each type.
                # This simulates the multi-file creation behavior of the daq software due to the max file size parameter
                movie_dest_file = get_sim_movie_dest(seqno)
                ph_dest_file = get_sim_ph_dest(seqno)
                simulated_data_files.extend([movie_dest_file, ph_dest_file])
                # logger.debug( f"Creating new simulated data files: {movie_dest_file=}, {ph_dest_file=}, {seqno=}, {fnum=}" )
                with open(movie_dest_file, "wb") as movie_dst, open(ph_dest_file, "wb") as ph_dst:
                    while not stop_io.is_set() and fnum < min(ph_nframes, movie_nframes):
                        # check if a new simulated file should be created
                        if int(fnum / frames_per_pff) > seqno:
                            seqno += 1
                            break
                        ph_data = ph_src.read(ph_frame_size)
                        ph_nbytes_written = ph_dst.write(ph_data)
                        ph_dst.flush()

                        movie_data = movie_src.read(movie_frame_size)
                        movie_nbytes_written = movie_dst.write(movie_data)

                        movie_dst.flush()

                        fnum += 1
                        time.sleep(update_interval)
                        if "early_exit" in kwargs and kwargs["early_exit"]:
                            early_exit_counter -= 1
                            if early_exit_counter == 0:
                                raise TimeoutError("test hp_io thread unexpected termination")
            if fnum >= min(ph_nframes, movie_nframes):
                logger.warning(f"simulated data acquisition reached EOF: {fnum=} >= {min(ph_nframes, movie_nframes)=}")
    except TimeoutError:
        pass
    finally:
        os.unlink(DAQ_ACTIVE_FILE)
        for file in simulated_data_files:
            os.unlink(file)
        logger.info("hp_sim thread exited")


def hp_io_thread_fn(
        data_dir: Path,
        module_id: int,
        dp_cfg: Dict[str, Any],
        update_interval: float,
        reader_states: List[Dict],
        stop_io: Event,
        valid: Event,
        logger: logging.Logger,
        simulate_daq: bool,
        **kwargs
) -> None:
    """ Receive pulse-height and movie-mode data from hashpipe and broadcast it to all active reader queues.
    Requires DAQ software to be active to properly initalize.
    """
    logger.info(f"Created a new hp_io thread with the following options: {kwargs=}")
    valid.clear()  # indicate hashpipe io channel is currently invalid
    # parse any kwargs
    if "early_exit_delay_seconds" in kwargs:
        early_exit_counter = kwargs["early_exit_delay_seconds"]
    else:
        early_exit_counter = 30
    try:
        def init_dp_cfg():
            """initialize dp_cfg with run-specific information and wait until hashpipe starts"""
            # check if a directory for [module_id] exists
            run_pattern = f"{data_dir}/module_{module_id}/obs_*"
            runs = glob(run_pattern)
            nruns = len(runs)
            if nruns == 0:
                raise FileNotFoundError(f'no run of module {module_id} in {run_pattern}')
            run_path = sorted(runs, key=os.path.getmtime)[-1]

            logger.info(f"{run_path=}")
            for dp in dp_cfg:
                # Get the current number of pff files with type [dp]
                dp_cfg[dp]['glob_pat'] = '%s/*%s*.pff' % (run_path, dp)
                # wait until hashpipe starts writing files
                nfiles = 0
                files = []
                while not stop_io.is_set() and nfiles == 0:
                    files = glob(dp_cfg[dp]['glob_pat'])
                    nfiles = len(files)
                    logger.debug(f'no file of type {dp} in {dp_cfg[dp]["glob_pat"]}')
                    time.sleep(0.5)
                if stop_io.is_set():
                    raise EnvironmentError("stop_io event is set")

                file = sorted(files, key=os.path.getmtime)[-1]
                filepath = file

                # wait until the filesize is large enough to read one image of type [dp]
                while not stop_io.is_set():
                    if os.path.getsize(filepath) >= dp_cfg[dp]['bytes_per_image']:
                        break
                    time.sleep(0.5)

                if stop_io.is_set():
                    raise EnvironmentError("stop_io event is set")

                # read the first frame of the file to determine the frame_size (this size is constant for the entire run)
                f = open(filepath, 'rb')
                logger.debug(f"{dp=}: {filepath=}: {dp_cfg[dp]['bytes_per_image']=}")
                frame_size = pff.img_frame_size(f, dp_cfg[dp]['bytes_per_image'])
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
                # check if a newer file for this data product has been created
                files = glob(d['glob_pat'])
                if len(files) > nfiles:
                    nfiles = len(files)
                    f.close()
                    file = sorted(files, key=os.path.getmtime)[-1]
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
            if not is_daq_active(simulate_daq):
                raise EnvironmentError("DAQ data flow stopped.")
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
                        module_id=module_id,
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
        logger.critical(f"hp_io thread encountered a fatal exception! '{repr(err)}'")
        # raise err
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
        self._hp_io_lock = threading.RLock()
        self._read_ok_condvar = threading.Condition(self._hp_io_lock)
        self._write_ok_condvar = threading.Condition(self._hp_io_lock)
        self._active_clients = {}  # dict of tid : {"client_ip":context.peer(), "thread": Thread} for debugging

        self._server_cfg = server_cfg

        # Create the server's logger
        self.logger = make_rich_logger(__name__, level=logging.INFO)

        # Load default hahspipe_io configuration
        with open(cfg_dir/self._server_cfg["default_hp_io_config_file"], "r") as f:
            self._hp_io_cfg = json.load(f)

        # State for single producer, multiple consumer hp_io access
        # A single IO thread manages the dataflow between multiple concurrent RPC threads and the hp_io thread:
        #   [single RPC writer -> hp_io thread] send messages to the hp_io thread
        #   [hp_io thread -> many RPC readers] broadcast image data to active read_queues
        self._hp_io_thread: Thread = None
        self._daq_sim_thread: Thread = None

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
        self._cancel_readers_event = Event()  # Causes all waiting and active reader RPCs to abort

        # Start the hp_io thread if server_cfg points to a valid hp_io_cfg
        if self._server_cfg["init_from_default"]:
            self.logger.info(f"Creating the initial hp_io thread from config: "
                             f"{self._server_cfg['init_from_default']=}")
            self._server_cfg['hp_io_init'] = True
            self._start_hp_io_thread(self._hp_io_cfg)
        else:
            self.logger.warning(f"An InitHpIo call is required to start the hp_io thread: "
                                f"{self._server_cfg['init_from_default']=}")

            self._server_cfg['hp_io_init'] = False

    def _cancel_all_readers(self):
        """Cancel all active and waiting reader RPCs."""
        self._cancel_readers_event.set()
        # signal any blocking readers to wake up and exit
        with self._hp_io_lock:
            self._read_ok_condvar.notify_all()
            for rs in [rs for rs in self._reader_states if rs['is_allocated']]:
                try:
                    rs['queue'].put_nowait("shutdown")
                except queue.Full:
                    pass

    def shutdown(self):
        self._shutdown_event.set()
        self._stop_io.set() # signal hp_io thread to exit gracefully
        self._cancel_all_readers()
        shutdown_record = dict()
        self._server_cfg['hp_io_init'] = False
        # wait for the hp_io thread to exit
        shutdown_record['stop_hp_io'] = self._stop_hp_io_thread(2)
        # wait for active server threads to exit
        for ac in self._active_clients.values():
            ac['thread'].join(timeout=1)
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
                lock_status_ok = False
        shutdown_record['lock_status_ok'] = lock_status_ok
        if all(shutdown_record.values()):
            self.logger.info("Successfully released all resources")
        else:
            self.logger.critical(f"Some server resources were not released: {shutdown_record=}")



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
        if not self._stop_hp_io_thread(timeout=self._hp_io_cfg['update_interval_seconds'] + 1):
            raise grpc.RpcError(grpc.StatusCode.INTERNAL, "Failed to terminate existing hp_io thread")
        self._stop_io.clear()

        dps = ["img16"]
        dp_cfg = self.get_dp_cfg(dps)
        data_dir = hp_io_cfg['data_dir']

        # Toggle simulation thread creation
        if hp_io_cfg['simulate_daq']:
            dps = ["img16", "ph256"]
            dp_cfg = self.get_dp_cfg(dps)
            data_dir = SIM_DATA_DIR
            self._daq_sim_thread = Thread(
                target=daq_sim_thread_fn,
                args=(
                    dp_cfg.copy(),
                    max(hp_io_cfg['update_interval_seconds'] * (2**0.5) / 1.5, self._server_cfg['min_hp_io_update_interval_seconds']),
                    self._stop_io,
                    self.logger
                ),
                kwargs={
                    "early_exit": True,  # causes the hp_io thread have a fatal exception after the given delay
                    "early_exit_delay_seconds": 25
                },
            )
            self._daq_sim_thread.start()

        # Create a new hp_io_thread using the client's configuration
        self._hp_io_thread = Thread(
            target=hp_io_thread_fn,
            args=(
                data_dir,
                1,
                dp_cfg.copy(),
                max(hp_io_cfg['update_interval_seconds'], self._server_cfg['min_hp_io_update_interval_seconds']),
                self._reader_states,
                self._stop_io,
                self._hp_io_valid,
                self.logger,
                hp_io_cfg['simulate_daq'],
            ),
            kwargs={
                "early_exit": False,  # causes the hp_io thread have a fatal exception after the given delay
                "early_exit_delay_seconds": 25
            },
            daemon=False,
        )
        self._hp_io_thread.start()


        # check if thread could be properly initialized
        self._hp_io_valid.wait(2)
        if self._is_hp_io_valid():
            self.logger.info("hp_io thread alive and valid")
            return True
        else:
            self._stop_hp_io_thread(1)
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
                if self._daq_sim_thread is not None:
                    self._daq_sim_thread.join(timeout)
            except RuntimeError as rerr:
                self.logger.critical(f"encountered runtime error while stopping hp_io thread: {rerr}")
                return False
            finally:
                if not self._hp_io_thread.is_alive():  # check if join succeeded or timeout happened while waiting
                    self.logger.info(f"Successfully terminated hp_io thread")
                    return True
        else:
            self.logger.debug("no hp_io thread to stop_io (doing nothing)")
            return True

    @contextmanager
    def _rw_lock_writer(self, context, force=False):
        tid = threading.get_ident()
        active = False
        try:
            with self._hp_io_lock:
                # BEGIN check-in critical section
                # All reader RPCs are long-lived server streaming operations.
                # The server's synchronization logic will prevent updates to _server_cfg while any reader RPCs are active,
                # so we should cancel any writer RPCs immediately

                if (not force) and self._rw_lock_state['ar'] > 0:
                    active_clients = str([c["client_ip"] for c in self._active_clients.values()])
                    emsg = (f"Cannot modify server state because there are {self._rw_lock_state['ar']} active "
                            f"streaming clients. Set force=True or stop the following clients and try again: {active_clients=}.")
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
                elif force and self._rw_lock_state['ar'] > 0:
                    self.logger.warning(f"Forcing server state modification despite active reader RPCs. ")
                    self._cancel_all_readers()

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
                # allow new readers to start waiting
                self._cancel_readers_event.clear()
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
                       not self._cancel_readers_event.is_set() and
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

                # check if reader RPCs are cancelled
                elif self._cancel_readers_event.is_set():
                    emsg = ("cancel_all_readers called during reader lock acquisition. "
                            "another client is likely configuring the server right now. "
                            "try again soon [skipping to check-out]")
                    self.logger.warning(emsg)
                    context.abort(grpc.StatusCode.CANCELLED, emsg)

                # check if the client is still active
                elif not context.is_active():
                    emsg = "client context terminated during reader lock acquisition [skipping to check-out]"
                    self.logger.error(emsg)
                    context.cancel()

                # check if the hp_io thread is valid
                elif self._server_cfg['hp_io_init'] and not self._is_hp_io_valid():
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

    def StreamImages(self, request, context):
        """Forward sample panoseti movie and pulse-height images to the client. [reader]"""
        self.logger.info(f"new StreamImages rpc from {urllib.parse.unquote(context.peer())}")
        # Validate request fields that don't require reading server state
        if not request.stream_movie_data and not request.stream_pulse_height_data:
            emsg = "At least one of the stream flags must be set to True"
            self.logger.info(emsg)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
        with self._rw_lock_reader(context) as reader_state:  # rid = allocated reader id for indexing into shared reader resources
            # BEGIN reader critical section
            # Validate request fields that require reading protected server state
            if not self._server_cfg['hp_io_init']:
                emsg = "Uninitialized hp_io thread. Run InitHpIo with a valid hp_io configuration to initialize it."
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)
            elif not (self._server_cfg["min_client_update_interval_seconds"]
                    <= request.update_interval_seconds
                    <= self._server_cfg['max_client_update_interval_seconds']):
                emsg = (f"update_interval_seconds must be in the interval "
                        f"[{self._server_cfg['min_client_update_interval_seconds']}, "
                        f"{self._server_cfg['max_client_update_interval_seconds']}"
                        f"seconds. Got {request.update_interval_seconds}")
                self.logger.critical(emsg)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)

            # Set stream filter options
            reader_state['config']['stream_movie_data'] = request.stream_movie_data
            reader_state['config']['stream_pulse_height_data'] = request.stream_pulse_height_data
            reader_state['config']['update_interval_seconds'] = request.update_interval_seconds

            # Clear old data from the read_queue
            rq = reader_state['queue']
            while not rq.empty():
                rq.get()

            # Valid server state -> start streaming!
            while context.is_active() and self._is_hp_io_valid() and not self._shutdown_event.is_set() and not self._cancel_readers_event.is_set():
                try:
                    # wait for next packet from the hp_io thread
                    # add a timeout to avoid starvation if hp_io thread unexpectedly exits while this thread is blocking on the read_queue
                    parsed_data = rq.get(timeout=request.update_interval_seconds)
                    if not isinstance(parsed_data, dict):
                        break
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
            if self._shutdown_event.is_set():
                emsg = "server shutdown initiated"
                context.abort(grpc.StatusCode.CANCELLED, emsg)
            elif self._cancel_readers_event.is_set():
                emsg = "cancel_all_readers: another client has likely forced a write to server state"
                context.abort(grpc.StatusCode.CANCELLED, emsg)
            elif not context.is_active():
                self.logger.info(f"StreamImages client disconnected")
            elif not self._stop_io.is_set():
                emsg = (f"The hp_io thread data stream unexpectedly became invalid! "
                        f"Check the server logs to debug this issue")
                self.logger.critical(emsg)
                context.abort(grpc.StatusCode.INTERNAL, emsg)
            else:
                context.abort(grpc.StatusCode.INTERNAL, "Unexpected error!")
            # END reader critical section

    def InitHpIo(self, request, context):
        """Initialize the hp_io thread with the given configuration. [writer]"""
        self.logger.info(f"new InitHpIo rpc from {urllib.parse.unquote(context.peer())}")

        # Validate request fields that don't require reading server state
        if (not request.simulate_daq) and (not os.path.exists(request.data_dir)):
            emsg = f"data_dir={request.data_dir} does not exist"
            self.logger.warning(emsg)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)

        # check if daq is active and real daq is being used. Note: simulated daq data flow always properly initialized
        if (not request.simulate_daq) and (not is_daq_active()):
            emsg = 'DAQ software is not active. Re-try hp_io thread creation once the daq software has been started.'
            self.logger.warning(emsg)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)

        # Step 1: read server state to validate init request
        with self._rw_lock_reader(context) as reader_state:
            # check if the requested update interval is not too short
            if request.update_interval_seconds < self._server_cfg['min_hp_io_update_interval_seconds']:
                emsg = (f"update_interval_seconds must be at least "
                        f"{self._server_cfg['min_hp_io_update_interval_seconds']} seconds. Got {request.update_interval_seconds}")
                self.logger.warning(emsg)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, emsg)

        self.logger.debug(f"(InitHpIo) passed validation checks")

        # attempt to change server state: modify hp_io thread
        with self._rw_lock_writer(context, force=request.force):
            self._server_cfg['hp_io_init'] = False
            last_hp_io_valid = self._is_hp_io_valid()
            stop_success = self._stop_hp_io_thread(timeout=self._hp_io_cfg['update_interval_seconds'] + 1)
            if not stop_success:
                emsg = "failed to stop hp_io thread!"
                self.logger.critical(emsg)
                context.abort(grpc.StatusCode.INTERNAL, emsg)
            self.logger.info("stopped existing hp_io thread")
            hp_io_cfg = {
                "data_dir": request.data_dir,
                "simulate_daq": request.simulate_daq,
                "update_interval_seconds": request.update_interval_seconds,
            }
            start_success = self._start_hp_io_thread(hp_io_cfg)
            if start_success:
                # commit client changes
                self.logger.info("InitHpIo transaction succeeded: new hp_io thread initialized")
                self._hp_io_cfg = hp_io_cfg
                self._server_cfg['hp_io_init'] = True
            else:
                # attempt to restart previously valid hp_io thread
                emsg = "failed to start hp_io thread."
                if last_hp_io_valid:
                    emsg += "Restarting hp_io with the previous configuration"
                    self._server_cfg['hp_io_init'] = self._start_hp_io_thread(self._hp_io_cfg)
                else:
                    emsg += "No previously valid hp_io thread to restart."
                self.logger.warning(emsg)

            return InitHpIoResponse(success=start_success)

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
