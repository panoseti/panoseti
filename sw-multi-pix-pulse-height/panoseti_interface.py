import os
import json
import mmap
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Union, Callable, Sequence
import warnings

import numpy as np
import matplotlib.pyplot as plt
from rich.logging import RichHandler
from rich.progress import Progress
import logging

# 1. Setup Rich Logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("PanosetiInterface")


@dataclass
class FrameConfig:
    """Holds structural information about a PFF file format."""
    header_size: int  # Bytes in the JSON header including newlines
    payload_size: int  # Bytes in the image data
    frame_size: int  # Total bytes per frame (header + marker + payload)
    image_shape: Tuple[int, int]
    dtype: np.dtype
    bytes_per_pixel: int
    format_name: str


class PFFSequence:
    """
    Represents a sequence of PFF files (a single data product run) as a continuous data stream.
    Handles split files (seqno_0, seqno_1...) and empty files robustly.
    """

    def __init__(self, file_paths: Sequence[Union[str, Path]]):
        self.file_paths = sorted([Path(p) for p in file_paths], key=lambda x: str(x))
        if not self.file_paths:
            raise ValueError("No file paths provided to PFFSequence.")

        self.name = self.file_paths[0].name.split('.seqno')[0]

        # Initialize metadata
        self.frame_config: Optional[FrameConfig] = None
        self._file_frame_counts: List[int] = []
        self._cumulative_frames: List[int] = []
        self._total_frames: int = 0

        # 2. Robust Structure Analysis (Handles empty first files)
        self._analyze_structure_robust()
        self._index_files()

    def _analyze_structure_robust(self) -> None:
        """
        Determines frame structure by inspecting the first non-empty file.
        """
        sample_file = None

        # Find first non-empty file
        for p in self.file_paths:
            if p.stat().st_size > 0:
                sample_file = p
                break

        if sample_file is None:
            logger.warning(f"All files in sequence {self.name} are empty (0 bytes).")
            return

        try:
            with open(sample_file, 'rb') as f:
                # Read a chunk large enough to contain the header
                # ph1024 headers are larger (~500 bytes), ph256 (~130 bytes).
                # Reading 4KB is safe.
                sample_chunk = f.read(4096)

                # Locate end of JSON header: ends with '}\n\n' followed by '*'
                # Note: The delimiter is strictly '\n\n' for the JSON block end in PFF specs
                header_end_match = re.search(b'}\n\n\\*', sample_chunk)

                if not header_end_match:
                    # Fallback for some older formats or malformed headers
                    header_end_match = re.search(b'\n\n\\*', sample_chunk)

                if not header_end_match:
                    raise ValueError(f"Could not find valid frame delimiter (}}\\n\\n*) in {sample_file}")

                # The '*' is at the end of the match
                marker_idx = header_end_match.end() - 1
                # The header is everything before the '*'
                header_bytes = sample_chunk[:marker_idx]

                try:
                    json_header = json.loads(header_bytes.decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON header: {e}")
                    raise

                self.header_size = len(header_bytes)

                # Determine Image Properties
                fname = sample_file.name

                # Parse BPP from filename or default to 2
                bpp = 2
                if 'bpp_' in fname:
                    bpp = int(fname.split('bpp_')[1].split('.')[0])

                # Parse format from filename
                fmt_name = 'unknown'
                if 'ph1024' in fname:
                    shape = (32, 32)
                    fmt_name = 'ph1024'
                elif 'ph256' in fname:
                    shape = (16, 16)
                    fmt_name = 'ph256'
                elif 'img16' in fname or 'img8' in fname:
                    shape = (32, 32)
                    fmt_name = 'img'
                else:
                    # Fallback assumption
                    shape = (32, 32)
                    logger.warning(f"Could not determine shape from filename {fname}, assuming 32x32.")

                pixels = shape[0] * shape[1]
                payload_size = pixels * bpp
                frame_size = self.header_size + 1 + payload_size  # +1 for '*' marker

                dtype = np.int16 if bpp == 2 else np.uint8

                self.frame_config = FrameConfig(
                    header_size=self.header_size,
                    payload_size=payload_size,
                    frame_size=frame_size,
                    image_shape=shape,
                    dtype=np.dtype(dtype),
                    bytes_per_pixel=bpp,
                    format_name=fmt_name
                )

                logger.info(
                    f"[green]Analyzed {fmt_name}[/green]: Frame Size={frame_size}b (Header={self.header_size}b, Payload={payload_size}b)")

        except Exception as e:
            logger.error(f"Failed to analyze structure for {sample_file}: {e}")
            raise

    def _index_files(self) -> None:
        """Calculates frame counts for all files, handling empty ones."""
        if not self.frame_config:
            # If config is missing (e.g. all empty files), we have 0 frames
            self._total_frames = 0
            self._file_frame_counts = [0] * len(self.file_paths)
            self._cumulative_frames = [0] * len(self.file_paths)
            return

        total = 0
        for p in self.file_paths:
            size = p.stat().st_size
            if size == 0:
                n_frames = 0
            else:
                n_frames = size // self.frame_config.frame_size
                remainder = size % self.frame_config.frame_size
                if remainder != 0:
                    logger.warning(f"File {p.name} has trailing bytes ({remainder}b). Truncated?")

            self._file_frame_counts.append(n_frames)
            self._cumulative_frames.append(total + n_frames)
            total += n_frames

        self._total_frames = total
        logger.info(f"Indexed sequence: {self._total_frames} frames across {len(self.file_paths)} files.")

    def __len__(self) -> int:
        return self._total_frames

    def _get_file_mapping(self, global_idx: int) -> Tuple[int, int]:
        """Maps global frame index -> (file_index, local_offset_index)."""
        if global_idx < 0: global_idx += self._total_frames
        if global_idx >= self._total_frames or global_idx < 0:
            raise IndexError(f"Index {global_idx} out of range (Total: {self._total_frames})")

        # Bisect is cleaner, but linear scan is fast enough for file lists < 1000 items
        for i, cutoff in enumerate(self._cumulative_frames):
            if global_idx < cutoff:
                prev_cutoff = self._cumulative_frames[i - 1] if i > 0 else 0
                return i, global_idx - prev_cutoff

        raise IndexError("Index out of range (Logic Error)")

    def iter_frames(self, start: int = 0, count: Optional[int] = None) -> Generator[
        Tuple[Dict, np.ndarray], None, None]:
        """
        Generator for iterating over frames efficiently.
        """
        if self._total_frames == 0:
            return

        end = self._total_frames if count is None else min(start + count, self._total_frames)
        current_idx = start

        while current_idx < end:
            file_idx, local_idx = self._get_file_mapping(current_idx)
            filepath = self.file_paths[file_idx]
            frames_in_file = self._file_frame_counts[file_idx]

            # Determine how many frames we can read from this file
            frames_to_read = min(end - current_idx, frames_in_file - local_idx)

            if frames_to_read <= 0:
                # Should not happen if logic is correct, but safe check
                current_idx += 1
                continue

            with open(filepath, 'rb') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    base_offset = local_idx * self.frame_config.frame_size

                    for _ in range(frames_to_read):
                        # Read Header
                        header_bytes = mm[base_offset: base_offset + self.frame_config.header_size]
                        # Read Data (Skip marker +1)
                        data_start = base_offset + self.frame_config.header_size + 1
                        data_bytes = mm[data_start: data_start + self.frame_config.payload_size]

                        try:
                            header = json.loads(header_bytes.decode('utf-8'))
                            arr = np.frombuffer(data_bytes, dtype=self.frame_config.dtype).reshape(
                                self.frame_config.image_shape)

                            yield header, arr
                        except Exception as e:
                            logger.error(f"Error reading frame {current_idx} in {filepath.name}: {e}")

                        base_offset += self.frame_config.frame_size
                        current_idx += 1

    def preview_frame(self, idx: int):
        """Plot a specific frame using Matplotlib."""
        if self._total_frames == 0:
            logger.warning("No frames to plot.")
            return

        header, img = list(self.iter_frames(start=idx, count=1))[0]

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(img, cmap='viridis')
        plt.colorbar(im, ax=ax, label='ADC / Pulse Height')
        ax.set_title(f"Frame {idx}\nModule: {header.get('quabo_num', '?')} Seq: {header.get('pkt_num', '?')}")
        plt.show()


# 4. Efficient Filter & Write Logic
def filter_pff_sequence(
        sequence: PFFSequence,
        output_filename: str,
        filter_func: Callable[[Dict, np.ndarray], bool]
) -> int:
    """
    Reads a PFFSequence, applies a filter, and writes passing frames to a new single PFF file.

    Args:
        sequence: The source PFFSequence.
        output_filename: Path to output .pff file.
        filter_func: Function accepting (header, image), returns True to keep.

    Returns:
        int: Number of frames written.
    """
    if sequence.frame_config is None:
        logger.warning("Source sequence has no valid configuration (empty?). Aborting filter.")
        return 0

    conf = sequence.frame_config
    written_count = 0

    # Pre-calculate separator bytes
    separator = b'*'
    newline_pad = b'\n\n'  # Header usually ends with this, but we ensure it

    logger.info(f"Starting filter job. Output: {output_filename}")

    with open(output_filename, 'wb') as f_out:
        with Progress() as progress:
            task = progress.add_task("[cyan]Filtering...", total=len(sequence))

            for header, img in sequence.iter_frames():
                if filter_func(header, img):
                    # 1. Update Sequence Numbers in Header (Optional but recommended)
                    # We might want to preserve original pkt_num for traceability,
                    # but typically 'seqno' implies file sequence.
                    # We leave the header content Mostly As-Is to preserve metadata.

                    # 2. Serialize Header
                    # We MUST ensure the header is exactly the same size as the original
                    # to maintain PFF random-access property if we want the output to be a valid PFF
                    # with the SAME config.

                    # Method: Re-dump JSON. If shorter, pad with spaces. If longer, we have a problem.
                    # PFF usually relies on fixed size headers.

                    json_str = json.dumps(header)
                    # The original header included the trailing \n\n.
                    # Let's target the exact byte size of the original header.

                    header_bytes = json_str.encode('utf-8')
                    current_len = len(header_bytes)
                    target_len = conf.header_size - 2  # -2 for the \n\n usually found at end

                    if current_len > target_len:
                        # This happens if we add keys or numbers get larger.
                        # Critical: PFF requires constant frame size for seek().
                        # If we can't fit it, we can't write a standard PFF without changing the spec.
                        # For now, we assume it fits or we truncate/warn.
                        logger.debug("Header grew larger than allocation. Truncating (risky).")
                        header_bytes = header_bytes[:target_len]

                    padding = b' ' * (target_len - len(header_bytes))

                    # Write Header + \n\n
                    f_out.write(header_bytes + padding + b'\n\n')

                    # Write Marker
                    f_out.write(separator)

                    # Write Data
                    f_out.write(img.tobytes())

                    written_count += 1

                progress.update(task, advance=1)

    logger.info(f"[green]Done![/green] Wrote {written_count} frames to {output_filename}")
    return written_count


class PanosetiRun:
    """Interface for a complete run directory."""

    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        self.products: Dict[str, PFFSequence] = {}
        self._scan()

    def _scan(self):
        # Group files by attributes, ignoring seqno and start time
        files_map = {}
        for f in self.run_dir.glob("*.pff"):
            # Example: start_2026...dp_ph1024.bpp_2.module_253.seqno_0.pff
            parts = f.name.split('.')
            # Key: dp_ph1024.bpp_2.module_253
            key_parts = [p for p in parts if not (p.startswith('start') or p.startswith('seqno') or p == 'pff')]
            key = ".".join(key_parts)

            if key not in files_map: files_map[key] = []
            files_map[key].append(f)

        for key, files in files_map.items():
            self.products[key] = PFFSequence(files)

    def get_product(self, product_name: str) -> PFFSequence:
        return self.products[product_name]

    def list_products(self):
        return list(self.products.keys())