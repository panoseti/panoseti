import json
import mmap
import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Sequence, Any
from pydantic import BaseModel, ValidationError

# Setup Rich Logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("PanosetiInterface")


# --- Pydantic Models for PFF Headers ---

class PFFHeader(BaseModel):
    """Base model for all PFF headers."""
    pkt_num: int
    pkt_tai: int
    pkt_nsec: int
    tv_sec: int
    tv_usec: int

    @property
    def timestamp(self) -> float:
        return self.tv_sec + self.tv_usec * 1e-6


class QuaboHeader(PFFHeader):
    quabo_num: int


class ModuleHeader(BaseModel):
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader


# --- Core Interface ---

class FrameConfig(BaseModel):
    header_size: int
    payload_size: int
    frame_size: int
    image_shape: Tuple[int, int]
    dtype_str: str
    bytes_per_pixel: int
    format_name: str
    padding_bytes: int = 1  # Separation bytes (e.g. '*')

    @property
    def dtype(self):
        return np.dtype(self.dtype_str)


class PFFSequence:
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

        # Cache for mmaps to avoid opening/closing constantly
        self._open_mmaps: Dict[int, mmap.mmap] = {}
        self._open_files: Dict[int, Any] = {}

        self._analyze_structure()
        self._index_files()

    def __del__(self):
        """Cleanup open file handles."""
        for mm in self._open_mmaps.values():
            mm.close()
        for f in self._open_files.values():
            f.close()

    def _parse_filename(self, fname: str) -> Dict[str, Any]:
        meta = {}
        parts = fname.split('.')
        for part in parts:
            if part.startswith('module_'):
                meta['module_id'] = int(part.split('_')[1])
            elif part.startswith('bpp_'):
                meta['bpp'] = int(part.split('_')[1])
            elif part.startswith('dp_'):
                meta['dp'] = part.split('_')[1]
        return meta

    def _analyze_structure(self):
        """Robustly determines frame structure from the first non-empty file."""
        sample_file = next((p for p in self.file_paths if p.stat().st_size > 0), None)
        if not sample_file:
            logger.debug(f"All files in {self.name} are empty.")
            return

        with open(sample_file, 'rb') as f:
            chunk = f.read(4096)
            match = re.search(b'}\n\n\\*', chunk)
            if not match:
                match = re.search(b'\n\n\\*', chunk)

            if not match:
                raise ValueError(f"Invalid PFF format in {sample_file}")

            # End of JSON part
            marker_idx = match.end() - 1
            # The separator is usually `*` (1 byte)

            header_bytes = chunk[:marker_idx]
            self.header_size = len(header_bytes)

            # Determine shape/type
            fmt = self.meta.get('dp', 'unknown')
            bpp = self.meta.get('bpp', 2)

            if 'ph1024' in fmt or 'img16' in fmt:
                shape = (32, 32)
                dtype = np.int16 if bpp == 2 else np.uint8
            elif 'ph256' in fmt:
                shape = (16, 16)
                dtype = np.int16
            else:
                shape = (32, 32)
                dtype = np.int16

            payload_size = shape[0] * shape[1] * bpp
            # Frame size = Header + Separator(1) + Payload
            frame_size = self.header_size + 1 + payload_size

            self.frame_config = FrameConfig(
                header_size=self.header_size,
                payload_size=payload_size,
                frame_size=frame_size,
                image_shape=shape,
                dtype_str=np.dtype(dtype).name,
                bytes_per_pixel=bpp,
                format_name=fmt
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
        """Returns a cached mmap for the given file index."""
        if file_idx in self._open_mmaps:
            return self._open_mmaps[file_idx]

        filepath = self.file_paths[file_idx]
        f = open(filepath, 'rb')
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

        self._open_files[file_idx] = f
        self._open_mmaps[file_idx] = mm
        return mm

    def get_frame(self, idx: int) -> Tuple[Union[QuaboHeader, ModuleHeader, Dict], np.ndarray]:
        """Retrieves a single frame (header + image)."""
        if idx >= self._total_frames or idx < 0:
            raise IndexError("Frame index out of range")

        # Locate File
        file_idx = 0
        local_idx = idx
        for i, limit in enumerate(self._cumulative_frames):
            if idx < limit:
                file_idx = i
                local_idx = idx - (self._cumulative_frames[i - 1] if i > 0 else 0)
                break

        conf = self.frame_config
        offset = local_idx * conf.frame_size

        # Use MMAP for reading
        mm = self._get_mmap(file_idx)

        # Read Header
        header_end = offset + conf.header_size
        header_bytes = mm[offset:header_end]

        # Parse Header
        try:
            header_dict = json.loads(header_bytes.decode('utf-8'))
            if 'quabo_0' in header_dict:
                header_obj = ModuleHeader(**header_dict)
            elif 'quabo_num' in header_dict:
                header_obj = QuaboHeader(**header_dict)
            else:
                header_obj = header_dict
        except (ValidationError, json.JSONDecodeError):
            header_obj = {}

        # Read Image
        img_start = header_end + 1  # Skip separator
        img_end = img_start + conf.payload_size

        img = np.frombuffer(mm[img_start:img_end], dtype=conf.dtype).reshape(conf.image_shape)
        return header_obj, img.copy()  # Return copy to be safe

    def get_image_array(self, start: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Efficiently retrieves a stack of images using strided mmap views.
        Skips headers without explicit read loops.
        """
        if count is None: count = self._total_frames - start
        count = min(count, self._total_frames - start)
        if count <= 0:
            return np.empty((0, *self.frame_config.image_shape), dtype=self.frame_config.dtype)

        chunks = []
        frames_collected = 0
        current_global = start

        conf = self.frame_config
        dtype = conf.dtype
        itemsize = dtype.itemsize

        # Pre-calculate strides for the numpy view
        # We want to view the file as (N_frames, H, W)
        # Stride 0 (Frame-to-Frame) = frame_size bytes
        # Stride 1 (Row-to-Row)     = Width * BPP
        # Stride 2 (Pixel-to-Pixel) = BPP

        # Note: This only works if frame_size is a multiple of itemsize.
        # e.g. if int16 (2 bytes), frame_size must be even.
        can_use_strided = (conf.frame_size % itemsize == 0)

        if can_use_strided:
            shape_strided = (conf.image_shape[0], conf.image_shape[1])
            strides_in_file = (
                conf.frame_size,  # Step to next frame
                shape_strided[1] * itemsize,  # Step to next row
                itemsize  # Step to next pixel
            )

        while frames_collected < count:
            # Locate file for current_global
            file_idx = 0
            for i, limit in enumerate(self._cumulative_frames):
                if current_global < limit:
                    file_idx = i
                    break

            # Calculate local range within this file
            prev_limit = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
            local_start = current_global - prev_limit
            file_total = self._file_frame_counts[file_idx]

            to_read = min(count - frames_collected, file_total - local_start)

            if to_read > 0:
                mm = self._get_mmap(file_idx)

                # Byte offset to the first pixel of the first requested frame
                # Offset = FrameStart + HeaderSize + Separator
                start_offset = (local_start * conf.frame_size) + conf.header_size + 1

                if can_use_strided:
                    # Create a strided view directly into the mmap
                    # This is zero-copy until we concatenate
                    chunk = np.ndarray(
                        shape=(to_read, *conf.image_shape),
                        dtype=dtype,
                        buffer=mm,
                        offset=start_offset,
                        strides=strides_in_file
                    )
                    chunks.append(chunk)
                else:
                    # Fallback: Iterative read if alignment is bad
                    # (Should rarely happen with standard PFF)
                    fallback_arr = np.empty((to_read, *conf.image_shape), dtype=dtype)
                    cursor = start_offset
                    for i in range(to_read):
                        end = cursor + conf.payload_size
                        fallback_arr[i] = np.frombuffer(mm[cursor:end], dtype=dtype).reshape(conf.image_shape)
                        cursor += conf.frame_size
                    chunks.append(fallback_arr)

            frames_collected += to_read
            current_global += to_read

        # Concatenate all chunks into a single contiguous array for the consumer
        # This is the only major memory copy operation.
        if len(chunks) == 1:
            return np.array(chunks[0])  # Force copy to detach from mmap if desired, or return chunks[0] if view is ok

        return np.concatenate(chunks, axis=0)

class PanosetiRun:
    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        self.products: Dict[str, PFFSequence] = {}

        # Load Configs Pydantic
        try:
            with open(self.run_dir / 'obs_config.json') as f:
                self.obs_config = json.load(f)
            with open(self.run_dir / 'quabo_uids.json') as f:
                self.quabo_uids = json.load(f)
        except Exception:
            logger.warning("Could not load full config context from run_obj dir.")
            self.obs_config = {}
            self.quabo_uids = {}

        self._scan()

    def _scan(self):
        files_map = {}
        for f in self.run_dir.glob("*.pff"):
            # Group by: dp_img16.bpp_2.module_253
            parts = f.name.split('.')
            key_parts = [p for p in parts if not (p.startswith('start') or p.startswith('seqno') or p == 'pff')]
            key = ".".join(key_parts)
            if key not in files_map: files_map[key] = []
            files_map[key].append(f)

        for k, v in files_map.items():
            try:
                self.products[k] = PFFSequence(v)
            except Exception as e:
                logger.warning(f"Skipping product {k}: {e}")

    def list_products(self):
        """Returns a list of available data products."""
        return [k for k, v in self.products.items() if len(v) > 0]

    def get_product(self, product_name: str) -> PFFSequence:
        """Returns the sequence object for a specific product key."""
        if product_name not in self.products:
            raise KeyError(f"Product {product_name} not found. Available: {self.list_products()}")
        return self.products[product_name]