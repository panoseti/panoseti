"""
PANOSETI File Interface
=======================

An optimized, object-oriented interface for reading PANOSETI observing runs.
Abstracts multi-file datasets into continuous data streams using memory mapping.

Dependencies:
    - numpy
    - pff (provided in the existing codebase)
"""

import os
import json
import mmap
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Generator
from dataclasses import dataclass
import numpy as np

# Assuming pff is available as a module, or we can inline the necessary util
import pff  # type: ignore


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class FrameLayout:
    """Stores the structural layout of a PFF frame."""
    header_size: int  # Bytes in the JSON header (including padding/newlines)
    image_offset: int  # Offset from start of frame to start of image data
    image_size: int  # Bytes in the binary image data
    frame_size: int  # Total bytes per frame
    bpp: int  # Bytes per pixel
    pixels: int  # Number of pixels (e.g., 1024)
    format: str  # numpy format string (e.g., 'H', 'B')


# -----------------------------------------------------------------------------
# Core Stream Class
# -----------------------------------------------------------------------------

class DataProductStream:
    """
    Represents a continuous stream of data for a specific Data Product (DP)
    and Module, spanning multiple physical files.

    Behaves like a read-only list/array of frames.
    """

    def __init__(self, file_paths: List[Union[str, Path]], dp_type: str, bpp: int):
        """
        Initialize the stream.

        Args:
            file_paths: List of sorted file paths making up this stream.
            dp_type: Data product type (e.g., 'img16', 'ph256').
            bpp: Bytes per pixel.
        """
        self.files = [Path(p) for p in file_paths]
        self.dp_type = dp_type
        self.bpp = bpp
        self._layout: Optional[FrameLayout] = None
        self._file_frame_counts: List[int] = []
        self._cumulative_frames: np.ndarray = np.array([])
        self._total_frames = 0

        # Resource management
        self._open_files_cache: Dict[int, Any] = {}
        self._mmaps_cache: Dict[int, mmap.mmap] = {}
        self._max_open_files = 8  # Keep a small buffer of open files

        if not self.files:
            raise ValueError("No files provided for DataProductStream.")

        self._analyze_structure()

    def _analyze_structure(self) -> None:
        """
        Determines frame size from the first file and indexes all files.
        """
        # 1. Determine Frame Layout from the first frame of the first file
        first_file = self.files[0]
        try:
            with open(first_file, 'rb') as f:
                # Robustly determine the size of the header and image
                # We assume the standard PFF format: JSON header + '*' + Binary Data

                # Mark start
                start_pos = f.tell()

                # Use pff.read_json logic but track bytes
                # PFF headers end with '\n\n'.
                # We read until we find the start of the image delimiter '*'
                # Note: This relies on the guarantee that headers are fixed size.

                header_str = pff.read_json(f)  # Consumes the JSON part
                if header_str is None:
                    raise ValueError(f"Empty or invalid first file: {first_file}")

                # After read_json, we should be at the '*' delimiter or close to it
                # pff.read_json might consume the newlines.
                # Let's verify position.
                current_pos = f.tell()
                header_bytes = current_pos - start_pos

                # Read delimiter
                delimiter = f.read(1)
                if delimiter != b'*':
                    raise ValueError(f"Expected '*' delimiter after JSON, found {delimiter}")

                # Determine image dimensions based on DP type
                # This logic mirrors pff.read_image but calculates size only
                pixels = 0
                dtype_char = ''

                if 'img' in self.dp_type:
                    # img16 = 32x32 = 1024 pixels
                    # img8 = 32x32 = 1024 pixels (usually)
                    pixels = 1024
                elif 'ph' in self.dp_type:
                    # ph256 = 16x16 = 256 pixels
                    # ph1024 = 32x32 = 1024 pixels
                    if '256' in self.dp_type:
                        pixels = 256
                    else:
                        pixels = 1024

                image_data_size = pixels * self.bpp

                # Verify we can read that many bytes
                f.seek(image_data_size, 1)  # Seek relative

                total_frame_size = header_bytes + 1 + image_data_size  # +1 for '*'

                # Set format for numpy
                if self.bpp == 1:
                    dtype_char = 'B'  # uint8
                elif self.bpp == 2:
                    dtype_char = 'H'  # uint16
                else:
                    dtype_char = 'B'  # Default fallback

                self._layout = FrameLayout(
                    header_size=header_bytes,
                    image_offset=header_bytes + 1,
                    image_size=image_data_size,
                    frame_size=total_frame_size,
                    bpp=self.bpp,
                    pixels=pixels,
                    format=dtype_char
                )

        except Exception as e:
            raise ValueError(f"Failed to analyze frame structure in {first_file}: {e}")

        # 2. Index all files
        # We assume all frames are the same size.
        counts = []
        for p in self.files:
            size = p.stat().st_size
            n_frames = size // self._layout.frame_size
            if size % self._layout.frame_size != 0:
                # Warning: File might be truncated or corrupt
                print(
                    f"Warning: File {p.name} size ({size}) is not a multiple of frame size ({self._layout.frame_size}). Truncating extra bytes.")
            counts.append(n_frames)

        self._file_frame_counts = counts
        self._cumulative_frames = np.cumsum([0] + counts)
        self._total_frames = self._cumulative_frames[-1]

    def _get_file_handle(self, file_idx: int) -> mmap.mmap:
        """
        Returns an mmap object for the specified file index, managing a simple LRU cache.
        """
        if file_idx in self._mmaps_cache:
            # Move to end (most recently used)
            val = self._mmaps_cache.pop(file_idx)
            self._mmaps_cache[file_idx] = val
            return val

        # Evict if full
        if len(self._mmaps_cache) >= self._max_open_files:
            oldest_idx = next(iter(self._mmaps_cache))
            self._mmaps_cache.pop(oldest_idx).close()
            self._open_files_cache.pop(oldest_idx).close()

        # Open new
        f = open(self.files[file_idx], 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self._open_files_cache[file_idx] = f
        self._mmaps_cache[file_idx] = mm
        return mm

    def _locate_frame(self, global_idx: int) -> Tuple[int, int]:
        """Maps a global frame index to (file_index, local_frame_index)."""
        if global_idx < 0:
            global_idx += self._total_frames

        if global_idx < 0 or global_idx >= self._total_frames:
            raise IndexError("Frame index out of range")

        # Binary search to find the file index
        # searchsorted returns the index where global_idx should be inserted to maintain order.
        # right side: side='right'. indices: [0, 100, 200]. val: 50 -> returns 1. File is 1-1=0.
        file_idx = np.searchsorted(self._cumulative_frames, global_idx, side='right') - 1
        local_idx = global_idx - self._cumulative_frames[file_idx]
        return file_idx, local_idx

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Dict, np.ndarray], List[Tuple[Dict, np.ndarray]]]:
        """
        Get frame(s) by index.
        Returns: (header_dict, image_array) or list of them.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total_frames)
            return [self.get_frame(i) for i in range(start, stop, step)]

        return self.get_frame(idx)

    def get_frame(self, idx: int) -> Tuple[Dict, np.ndarray]:
        """Retrieve a specific frame (Header + Image)."""
        file_idx, local_idx = self._locate_frame(idx)
        mm = self._get_file_handle(file_idx)

        start_byte = local_idx * self._layout.frame_size

        # Read Header
        # We know exact header size
        json_bytes = mm[start_byte: start_byte + self._layout.header_size]
        try:
            header = json.loads(json_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            # Fallback for potentially malformed padding
            header = json.loads(json_bytes.decode('utf-8').rstrip('\x00'))

        # Read Image
        img_start = start_byte + self._layout.image_offset
        img_end = img_start + self._layout.image_size
        raw_bytes = mm[img_start: img_end]

        # Convert to numpy
        # Note: frombuffer is zero-copy (shares memory with mmap)
        # We usually want a copy if we are closing mmaps frequently,
        # but here the LRU keeps it alive. To be safe for the user, we copy.
        img_arr = np.frombuffer(raw_bytes, dtype=self._layout.format).copy()

        # Reshape
        dim = int(np.sqrt(self._layout.pixels))
        img_arr = img_arr.reshape((dim, dim))

        return header, img_arr

    def get_image(self, idx: int) -> np.ndarray:
        """Optimized fetch for just the image data (skips JSON parsing)."""
        file_idx, local_idx = self._locate_frame(idx)
        mm = self._get_file_handle(file_idx)

        offset = (local_idx * self._layout.frame_size) + self._layout.image_offset
        raw_bytes = mm[offset: offset + self._layout.image_size]

        dim = int(np.sqrt(self._layout.pixels))
        return np.frombuffer(raw_bytes, dtype=self._layout.format).copy().reshape((dim, dim))

    def close(self):
        """Clean up file handles."""
        for mm in self._mmaps_cache.values():
            mm.close()
        for f in self._open_files_cache.values():
            f.close()
        self._mmaps_cache.clear()
        self._open_files_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -----------------------------------------------------------------------------
# Observing Run Interface
# -----------------------------------------------------------------------------

class ObservingRun:
    """
    Main interface to a PANOSETI Observing Run directory.
    Automatically organizes files by module and data product.
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        self.config_files: Dict[str, Any] = {}
        self.modules: List[int] = []
        self.data_products: Dict[str, Dict[int, DataProductStream]] = {}
        # Structure: { 'img16': { module_id: DataStream, ... }, ... }

        self._scan_directory()
        self._load_configs()

    def _load_configs(self):
        """Loads static JSON config files found in the directory."""
        for json_file in self.run_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    self.config_files[json_file.stem] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config {json_file}: {e}")

    def _scan_directory(self):
        """Scans PFF files and groups them."""
        # Regex to parse filenames like:
        # start_2022-02-01T07:30:08Z.dp_img16.bpp_2.dome_0.module_14.seqno_5.pff
        # Note: Adapting regex based on provided parse_name examples

        files_by_key = {}  # Key: (dp, module, bpp), Value: list of (seqno, filepath)

        for fpath in self.run_dir.glob("*.pff"):
            fname = fpath.name
            if fname == 'hk.pff':
                continue  # Handle HK separately if needed

            # Using pff.parse_name or custom logic
            try:
                # Custom parse to avoid dependency issues if pff.py changes
                parts = fname.split('.')
                meta = {}
                for part in parts:
                    if '_' in part:
                        k, v = part.split('_', 1)
                        meta[k] = v

                if 'dp' in meta and 'module' in meta and 'seqno' in meta:
                    dp = meta['dp']
                    mod = int(meta['module'])
                    seq = int(meta['seqno'])
                    bpp = int(meta.get('bpp', 2))  # Default to 2 if missing?

                    key = (dp, mod, bpp)
                    if key not in files_by_key:
                        files_by_key[key] = []
                    files_by_key[key].append((seq, fpath))
            except Exception:
                continue

        # Create Streams
        self.modules = sorted(list(set(k[1] for k in files_by_key.keys())))

        for (dp, mod, bpp), file_list in files_by_key.items():
            # Sort by sequence number
            sorted_files = [x[1] for x in sorted(file_list, key=lambda x: x[0])]

            if dp not in self.data_products:
                self.data_products[dp] = {}

            print(f"Initializing stream: {dp} Module {mod} ({len(sorted_files)} files)...")
            try:
                stream = DataProductStream(sorted_files, dp, bpp)
                self.data_products[dp][mod] = stream
            except Exception as e:
                print(f"Error initializing stream for {dp} Mod {mod}: {e}")

    def get_stream(self, data_product: str, module_id: int) -> Optional[DataProductStream]:
        """
        Get the data stream for a specific product and module.
        Example: run.get_stream('img16', 14)
        """
        return self.data_products.get(data_product, {}).get(module_id)

    def close(self):
        """Close all open streams."""
        for dp_map in self.data_products.values():
            for stream in dp_map.values():
                stream.close()


# -----------------------------------------------------------------------------
# Usage Example (Augmenting io.py)
# -----------------------------------------------------------------------------
# You can append the following logic to your existing io.py or use it alongside.

def example_usage():
    run_path = "./obs_Lick.start_2023..."

    # Initialize the run
    run = ObservingRun(run_path)

    # Access a stream (e.g., img16 from module 14)
    stream = run.get_stream('img16', 14)

    if stream:
        print(f"Total Frames: {len(stream)}")

        # Random access (fetches from correct file automatically)
        header, img = stream[5000]
        print(f"Frame 5000 Time: {header.get('pkt_tai')}")
        print(f"Image Mean: {np.mean(img)}")

        # Fast image-only access
        img_only = stream.get_image(5001)

        # Slicing
        frames = stream[0:10]  # Returns first 10 frames

    run.close()


if __name__ == "__main__":
    pass