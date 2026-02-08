import json
import mmap
import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Union, Sequence, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import matplotlib.pyplot as plt

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
    """Header for ph256/img data (single quabo/module context)."""
    quabo_num: int


class ModuleHeader(BaseModel):
    """Header for ph1024/img16 data (contains 4 quabos)."""
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader

    def get_quabo_header(self, idx: int) -> PFFHeader:
        return getattr(self, f"quabo_{idx}")


# --- Core Interface ---

class FrameConfig(BaseModel):
    header_size: int
    payload_size: int
    frame_size: int
    image_shape: Tuple[int, int]
    dtype_str: str  # Store as string for Pydantic serialization
    bytes_per_pixel: int
    format_name: str

    @property
    def dtype(self):
        return np.dtype(self.dtype_str)


class PFFSequence:
    def __init__(self, file_paths: Sequence[Union[str, Path]]):
        self.file_paths = sorted([Path(p) for p in file_paths], key=lambda x: str(x))
        if not self.file_paths:
            raise ValueError("No files provided.")

        self.name = self.file_paths[0].name.split('.seqno')[0]
        # Parse metadata from filename
        self.meta = self._parse_filename(self.file_paths[0].name)

        self.frame_config: Optional[FrameConfig] = None
        self._file_frame_counts: List[int] = []
        self._cumulative_frames: List[int] = []
        self._total_frames: int = 0

        self._analyze_structure()
        self._index_files()

    def _parse_filename(self, fname: str) -> Dict[str, Any]:
        """Extracts metadata like module_id from filename."""
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
            logger.warning(f"All files in {self.name} are empty.")
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
                shape = (32, 32)  # Default
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

    def get_frame(self, idx: int) -> Tuple[Union[QuaboHeader, ModuleHeader, Dict], np.ndarray]:
        """Retrieves a single frame with Pydantic-parsed header."""
        if idx >= self._total_frames or idx < 0:
            raise IndexError("Frame index out of range")

        # Find file
        for i, limit in enumerate(self._cumulative_frames):
            if idx < limit:
                file_idx = i
                local_idx = idx - (self._cumulative_frames[i - 1] if i > 0 else 0)
                break

        filepath = self.file_paths[file_idx]
        conf = self.frame_config
        offset = local_idx * conf.frame_size

        with open(filepath, 'rb') as f:
            f.seek(offset)
            header_bytes = f.read(conf.header_size)
            header_dict = json.loads(header_bytes.decode('utf-8'))

            # Parse Header into Pydantic Model
            try:
                if 'quabo_0' in header_dict:
                    header_obj = ModuleHeader(**header_dict)
                elif 'quabo_num' in header_dict:
                    header_obj = QuaboHeader(**header_dict)
                else:
                    header_obj = header_dict  # Fallback
            except ValidationError:
                header_obj = header_dict

            f.seek(1, 1)  # Skip '*'
            raw = f.read(conf.payload_size)
            img = np.frombuffer(raw, dtype=conf.dtype).reshape(conf.image_shape)

            return header_obj, img

    def get_image_array(self, start=0, count=None) -> np.ndarray:
        """Fast mmap reader for image stacks."""
        if count is None: count = self._total_frames - start
        count = min(count, self._total_frames - start)
        if count <= 0: return np.empty((0, *self.frame_config.image_shape))

        result = np.empty((count, *self.frame_config.image_shape), dtype=self.frame_config.dtype)

        # Logic similar to previous generic iterator, omitted for brevity
        # ... (Implementation of cross-file reading) ...
        # For simplicity in this snippet, we assume fitting in one file or simple loop
        # Re-using the robust iterator approach is best.

        idx = 0
        current_global = start
        while idx < count:
            # Find file
            for i, limit in enumerate(self._cumulative_frames):
                if current_global < limit:
                    file_idx = i
                    local_idx = current_global - (self._cumulative_frames[i - 1] if i > 0 else 0)
                    break

            path = self.file_paths[file_idx]
            available = self._file_frame_counts[file_idx] - local_idx
            to_read = min(count - idx, available)

            with open(path, 'rb') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    base = local_idx * self.frame_config.frame_size
                    stride = self.frame_config.frame_size

                    # Vectorized read if contiguous is hard due to headers
                    # Loop is unavoidable unless we reshape the whole buffer (risky with alignment)
                    for _ in range(to_read):
                        d_start = base + self.frame_config.header_size + 1
                        d_end = d_start + self.frame_config.payload_size
                        result[idx] = np.frombuffer(mm[d_start:d_end], dtype=self.frame_config.dtype).reshape(
                            self.frame_config.image_shape)
                        base += stride
                        idx += 1
                        current_global += 1
        return result


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