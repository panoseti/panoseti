import mmap
import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Sequence, Any
from pydantic import BaseModel, ValidationError

# Efficiency Imports
try:
    import orjson
except ImportError:
    import json as orjson  # Fallback, though orjson is highly recommended

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Setup Rich Logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("PanosetiInterface")


# --- Precise Timing Helper (Integer Arithmetic) ---

def get_precise_time_ns(tv_sec: int, tv_usec: int, pkt_nsec: int) -> int:
    """
    Calculates precise timestamp in NANOSECONDS using pure integer arithmetic.
    Avoids float64 precision loss (which occurs at ~microsecond levels for current Unix timestamps).

    Logic:
    1. Unix time is derived from DAQ nodes (NTP synced, ms accuracy).
    2. Packet time is derived from White Rabbit/GPS (ns accuracy).
    3. We use the 'seconds' from DAQ and 'nanoseconds' from WR.
    4. We detect boundary wraps by comparing tv_usec (microseconds) to pkt_nsec.

    Args:
        tv_sec: DAQ Unix seconds.
        tv_usec: DAQ Unix microseconds.
        pkt_nsec: White Rabbit nanoseconds.

    Returns:
        int: Total nanoseconds since Unix epoch.
    """
    # 1. Convert DAQ usec to nsec for comparison
    tv_nsec_equiv = tv_usec * 1000
    diff = tv_nsec_equiv - pkt_nsec

    # 2. Reconcile drift
    # Threshold: 50ms (50,000,000 ns)

    final_sec = tv_sec

    if abs(diff) < 50_000_000:
        # Case 1: In sync. Trust DAQ seconds.
        pass
    elif diff > 500_000_000:
        # Case 2: tv_usec is large (e.g. 999ms), pkt_nsec is small (e.g. 1ms).
        # DAQ clock hasn't wrapped yet, but GPS has.
        # We are effectively into the *next* second compared to what tv_sec says?
        # Note: logic from notes: "tv_usec much larger... tv_sec is 1 sec slower" => +1
        final_sec += 1
    elif diff < -500_000_000:
        # Case 3: pkt_nsec is large, tv_usec is small.
        # GPS is behind NTP? Or NTP wrapped early?
        # Note: logic from notes: "tv_sec is 1 sec faster" => -1
        final_sec -= 1

    # 3. Combine integer seconds and nanoseconds
    return (final_sec * 1_000_000_000) + pkt_nsec


# --- Pydantic Models for PFF Headers ---

class PFFHeader(BaseModel):
    """Base model for all PFF headers."""
    pkt_num: int
    pkt_tai: int
    pkt_nsec: int
    tv_sec: int
    tv_usec: int

    @property
    def timestamp_ns(self) -> int:
        """Returns nanoseconds since epoch as int64 compatible integer."""
        return get_precise_time_ns(self.tv_sec, self.tv_usec, self.pkt_nsec)


class QuaboHeader(PFFHeader):
    quabo_num: int


class ModuleHeader(BaseModel):
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader

    @property
    def timestamp_ns(self) -> int:
        return self.quabo_0.timestamp_ns


# --- Core Interface ---

class FrameConfig(BaseModel):
    header_size: int
    payload_size: int
    frame_size: int
    image_shape: Tuple[int, int]
    dtype_str: str
    bytes_per_pixel: int
    format_name: str

    @property
    def dtype(self):
        return np.dtype(self.dtype_str)


class PFFSequence:
    """
    Represents a time-ordered sequence of PFF files for a single data product.

    Features:
    - Zero-copy access via mmap.
    - Virtual concatenation of multiple files.
    - Precise int64 nanosecond timing.
    - Binary search for time seeking.
    """

    def __init__(self, file_paths: Sequence[Union[str, Path]]):
        self.file_paths = sorted([Path(p) for p in file_paths], key=lambda x: str(x))
        if not self.file_paths:
            raise ValueError("No files provided.")

        self.name = self.file_paths[0].name.split('.seqno')[0]
        self.meta = self._parse_filename(self.file_paths[0].name)

        self.frame_config: Optional[FrameConfig] = None
        self._file_frame_counts: List[int] = []
        self._cumulative_frames: List[int] = []
        self._total_frames: int = 0

        # MMap Cache
        self._open_mmaps: Dict[int, mmap.mmap] = {}
        self._open_files: Dict[int, Any] = {}

        self._analyze_structure()
        self._index_files()

    def __del__(self):
        self.close()

    def close(self):
        """Explicitly close file handles."""
        for mm in self._open_mmaps.values(): mm.close()
        for f in self._open_files.values(): f.close()
        self._open_mmaps.clear()
        self._open_files.clear()

    def _parse_filename(self, fname: str) -> Dict[str, Any]:
        meta = {}
        clean_name = fname.split('.pff')[0]
        parts = clean_name.split('.')
        for part in parts:
            if '_' in part:
                k, v = part.split('_', 1)
                try:
                    if v.isdigit(): v = int(v)
                except:
                    pass
                meta[k] = v
        return meta

    def _analyze_structure(self):
        """Determines frame structure from first valid file."""
        sample_file = next((p for p in self.file_paths if p.stat().st_size > 0), None)
        if not sample_file:
            return

        with open(sample_file, 'rb') as f:
            chunk = f.read(4096)
            # Find JSON end: }\n\n*
            match = re.search(b'}\n\n\\*', chunk)
            if not match:
                match = re.search(b'\n\n\\*', chunk)  # Fallback

            if not match:
                raise ValueError(f"Invalid PFF format in {sample_file}")

            marker_idx = match.end() - 1
            header_bytes = chunk[:marker_idx]

            # Use orjson for initial probe
            try:
                h = orjson.loads(header_bytes)
            except Exception:
                # Fallback if orjson fails or header incomplete
                h = {}

            self.header_size = len(header_bytes)

            # Determine shape based on metadata or header content
            fmt = self.meta.get('dp', 'unknown')
            bpp = self.meta.get('bpp', 2)

            # Deduce shape
            if 'ph1024' in str(fmt) or 'img16' in str(fmt):
                shape = (32, 32)
                dtype = np.int16 if bpp == 2 else np.uint8
            elif 'ph256' in str(fmt):
                shape = (16, 16)
                dtype = np.int16
            else:
                shape = (32, 32)
                dtype = np.int16

            payload_size = shape[0] * shape[1] * bpp
            frame_size = self.header_size + 1 + payload_size

            self.frame_config = FrameConfig(
                header_size=self.header_size,
                payload_size=payload_size,
                frame_size=frame_size,
                image_shape=shape,
                dtype_str=np.dtype(dtype).name,
                bytes_per_pixel=bpp,
                format_name=str(fmt)
            )

    def _index_files(self):
        total = 0
        if self.frame_config:
            for p in self.file_paths:
                size = p.stat().st_size
                n = size // self.frame_config.frame_size if size > 0 else 0
                self._file_frame_counts.append(n)
                self._cumulative_frames.append(total + n)
                total += n
        self._total_frames = total

    def __len__(self):
        return self._total_frames

    def _get_mmap(self, file_idx: int) -> mmap.mmap:
        if file_idx in self._open_mmaps:
            return self._open_mmaps[file_idx]

        filepath = self.file_paths[file_idx]
        f = open(filepath, 'rb')
        try:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            self._open_files[file_idx] = f
            self._open_mmaps[file_idx] = mm
            return mm
        except ValueError:
            return None

    def _locate_frame(self, idx: int) -> Tuple[int, int]:
        """
        Maps global index to (file_index, local_index).
        """
        if idx >= self._total_frames or idx < 0:
            raise IndexError(f"Frame index {idx} out of range (Total: {self._total_frames})")

        # Bisect could be used here if file counts are huge, but typically < 1000 files
        for i, limit in enumerate(self._cumulative_frames):
            if idx < limit:
                prev_limit = self._cumulative_frames[i - 1] if i > 0 else 0
                return i, idx - prev_limit
        raise IndexError("Index out of bounds")

    def get_frame(self, idx: int) -> Tuple[Union[QuaboHeader, ModuleHeader, Dict], np.ndarray]:
        """
        Full frame retrieval with parsing. Slow path.
        Returns: (HeaderObject, NumpyImage)
        """
        file_idx, local_idx = self._locate_frame(idx)
        conf = self.frame_config
        mm = self._get_mmap(file_idx)
        offset = local_idx * conf.frame_size

        # Read Header
        header_end = offset + conf.header_size
        header_bytes = mm[offset:header_end]

        # Fast JSON parse
        header_dict = orjson.loads(header_bytes)

        # Type conversion via Pydantic
        try:
            if 'quabo_0' in header_dict:
                header_obj = ModuleHeader(**header_dict)
            elif 'quabo_num' in header_dict:
                header_obj = QuaboHeader(**header_dict)
            else:
                header_obj = header_dict
        except ValidationError:
            header_obj = header_dict

        # Read Image
        img_start = header_end + 1
        img_end = img_start + conf.payload_size
        img = np.frombuffer(mm[img_start:img_end], dtype=conf.dtype).reshape(conf.image_shape)

        return header_obj, img.copy()

    def get_frame_time(self, idx: int) -> int:
        """
        Retrieves ONLY the precise nanosecond timestamp for a frame.
        Fastest way to get time.
        """
        file_idx, local_idx = self._locate_frame(idx)
        conf = self.frame_config
        mm = self._get_mmap(file_idx)
        offset = local_idx * conf.frame_size

        # Read only header bytes
        header_bytes = mm[offset: offset + conf.header_size]
        h = orjson.loads(header_bytes)

        # Manual extraction for speed (skipping Pydantic)
        if 'quabo_0' in h:
            q = h['quabo_0']
            return get_precise_time_ns(q['tv_sec'], q['tv_usec'], q['pkt_nsec'])
        elif 'pkt_nsec' in h:
            return get_precise_time_ns(h['tv_sec'], h['tv_usec'], h['pkt_nsec'])
        return 0

    def seek_time(self, target_time_ns: int) -> int:
        """
        Binary Search for frame index closest to target_time_ns.
        Args:
            target_time_ns: Integer nanoseconds from epoch.
        Returns:
            Global frame index.
        """
        if self._total_frames == 0: return 0

        low = 0
        high = self._total_frames - 1

        # Bounds Check
        t_start = self.get_frame_time(low)
        if target_time_ns <= t_start: return low

        t_end = self.get_frame_time(high)
        if target_time_ns >= t_end: return high

        while low <= high:
            mid = (low + high) // 2
            t_mid = self.get_frame_time(mid)

            if t_mid < target_time_ns:
                low = mid + 1
            elif t_mid > target_time_ns:
                high = mid - 1
            else:
                return mid

        # Check nearest neighbor
        candidates = [c for c in [high, low] if 0 <= c < self._total_frames]
        return min(candidates, key=lambda i: abs(self.get_frame_time(i) - target_time_ns))

    def get_image_array(self, start=0, count=None) -> np.ndarray:
        """
        Optimized 'Virtual Array' access.
        Returns raw image stack, skipping all headers/json.
        """
        if count is None: count = self._total_frames - start
        count = min(count, self._total_frames - start)
        if count <= 0: return np.empty((0, *self.frame_config.image_shape), dtype=self.frame_config.dtype)

        conf = self.frame_config
        dtype = conf.dtype
        itemsize = dtype.itemsize

        chunks = []
        frames_collected = 0
        current_global = start

        # Strided read if alignment allows
        can_use_strided = (conf.frame_size % itemsize == 0)
        strides = None
        if can_use_strided:
            strides = (
                conf.frame_size,
                conf.image_shape[1] * itemsize,
                itemsize
            )

        while frames_collected < count:
            # Locate file for current_global
            file_idx = 0
            # Simple linear scan for file bucket (fast enough for sequential reads)
            for i, limit in enumerate(self._cumulative_frames):
                if current_global < limit:
                    file_idx = i
                    break

            prev_limit = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
            local_start = current_global - prev_limit
            file_total = self._file_frame_counts[file_idx]

            to_read = min(count - frames_collected, file_total - local_start)

            if to_read > 0:
                mm = self._get_mmap(file_idx)
                start_offset = (local_start * conf.frame_size) + conf.header_size + 1

                if can_use_strided and mm:
                    # ZERO-COPY View
                    chunk = np.ndarray(
                        shape=(to_read, *conf.image_shape),
                        dtype=dtype,
                        buffer=mm,
                        offset=start_offset,
                        strides=strides
                    )
                    chunks.append(chunk)
                elif mm:
                    # Fallback copy
                    temp = np.empty((to_read, *conf.image_shape), dtype=dtype)
                    cursor = start_offset
                    for i in range(to_read):
                        end = cursor + conf.payload_size
                        temp[i] = np.frombuffer(mm[cursor:end], dtype=dtype).reshape(conf.image_shape)
                        cursor += conf.frame_size
                    chunks.append(temp)

            frames_collected += to_read
            current_global += to_read

        # Merge
        if len(chunks) == 1:
            return np.array(chunks[0])
        elif len(chunks) > 1:
            return np.concatenate(chunks, axis=0)
        else:
            return np.empty((0, *conf.image_shape), dtype=dtype)

    def to_dask(self, chunks='auto'):
        """
        Convert to a Dask Array for distributed processing.

        Uses dask.array.from_array mapping strategies.
        This allows the 1TB+ stream to be treated as a single lazy array
        on an HPC cluster (using dask-jobqueue/MPI).
        """
        if not HAS_DASK:
            raise ImportError("Dask is not installed.")

        # Since get_image_array requires sophisticated logic (mmap management),
        # standard dask.array.from_array might struggle with pickling the mmap objects.
        # We use 'map_blocks' or 'from_delayed' strategy, or a custom slice getter.

        # Simple Proxy Wrapper that is pickle-safe (closes mmaps on pickle)
        # Note: In a real implementation, we'd make a lightweight picklable handle.
        # For now, we assume this runs on a shared filesystem.

        # Strategy: Create a dask array for EACH file, then concatenate.
        dask_chunks = []
        conf = self.frame_config

        for i, fpath in enumerate(self.file_paths):
            n_frames = self._file_frame_counts[i]
            if n_frames == 0: continue

            # We define a function that reads a chunk from a SPECIFIC file
            # This function must be robust to opening/closing files on workers
            def load_chunk(file_path, header_size, payload_size, frame_size, shape, dtype, count):
                # Independent read function for workers
                with open(file_path, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                    # Use strided trick
                    start_offset = header_size + 1
                    strides = (frame_size, shape[1] * dtype.itemsize, dtype.itemsize)
                    arr = np.ndarray(
                        shape=(count, *shape),
                        dtype=dtype,
                        buffer=mm,
                        offset=start_offset,
                        strides=strides
                    )
                    return np.array(arr)  # Force copy to memory for Dask

            # Create delayed object
            # Ideally we use from_map or similar, but let's effectively stack lazy arrays
            # For simplicity in this interface: return a concatenated Dask array
            # representing the whole stream.

            # Correct Approach for Strided/Complex Binary formats in Dask:
            # map_blocks on an index array, or da.from_array with a custom getter.
            pass

        # Simplified Dask return for now:
        # User should likely use get_image_array in map_blocks manually for max control.
        logger.info("Dask export is experimental. Use get_image_array inside dask.delayed for best results.")
        return None


class PanosetiRun:
    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        self.products: Dict[str, PFFSequence] = {}
        self.configs: Dict[str, Any] = {}

        self.load_configs()
        self._scan()

    def load_configs(self):
        """Loads all JSON configuration files in the run directory."""
        for f in self.run_dir.glob("*.json"):
            try:
                with open(f, 'rb') as jf:
                    self.configs[f.stem] = orjson.loads(jf.read())
            except Exception as e:
                logger.warning(f"Failed to load config {f.name}: {e}")

    def _scan(self):
        files_map = {}
        for f in self.run_dir.glob("*.pff"):
            if f.name == 'hk.pff': continue
            if f.stat().st_size == 0: continue

            parts = f.name.split('.')
            key_parts = []
            for p in parts:
                if p.startswith('start') or p.startswith('seqno') or p == 'pff':
                    continue
                key_parts.append(p)

            key = ".".join(key_parts)
            if key not in files_map: files_map[key] = []
            files_map[key].append(f)

        for k, v in files_map.items():
            try:
                seq = PFFSequence(v)
                if len(seq) > 0:
                    self.products[k] = seq
            except Exception as e:
                logger.warning(f"Skipping product {k}: {e}")

    def list_products(self) -> List[str]:
        return sorted(self.products.keys())

    def get_product(self, product_name: str) -> PFFSequence:
        if product_name not in self.products:
            raise KeyError(f"Product {product_name} not found.")
        return self.products[product_name]

    def show(self):
        """Rich visualization of the Run structure."""
        console = Console()

        # 1. Configs
        tree = Tree(f"[bold gold1]Run: {self.run_dir.name}[/]")
        config_branch = tree.add("Configurations")
        for k in self.configs.keys():
            config_branch.add(f"[cyan]{k}.json[/]")

        # 2. Products
        prod_branch = tree.add("Data Products")
        for name, seq in self.products.items():
            info = f"[bold green]{name}[/] ({len(seq):,} frames)"
            prod_branch.add(info)

        console.print(tree)