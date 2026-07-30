# functions to parse PFF files,
# and to create and parse PFF dir/file names

from __future__ import annotations

import datetime
import json
import os
import struct
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, BinaryIO

import numpy as np


# returns the string (doesn't parse it)
#
def read_json(f: BinaryIO) -> str | None:
    c = f.read(1)
    if c ==b'':
        return None
    if c != b'{':
        raise Exception('read_json(): expected {, got', c)
    s = '{'
    last_nl = False
    while True:
        c = f.read(1)
        if c == b'\n':
            if last_nl:
                break
            last_nl = True
        else:
            last_nl = False
        s += c.decode()
    return s

# returns the image as a list of N numbers
# see https://docs.python.org/3/library/struct.html
#
def read_image(f: BinaryIO, img_size: int, bytes_per_pixel: int) -> list[int] | None:
    c = f.read(1)
    if c == b'':
        return None
    if c != b'*':
        raise Exception('bad type code')
    if img_size == 32:
        if bytes_per_pixel == 2:
            return list(struct.unpack("1024H", f.read(2048)))
        elif bytes_per_pixel == 1:
            return list(struct.unpack("1024B", f.read(1024)))
        else:
            raise Exception(f"bad bytes per pixel: {bytes_per_pixel}")
    elif img_size == 16:
        if bytes_per_pixel == 2:
            return list(struct.unpack("256H", f.read(512)))
        elif bytes_per_pixel == 2:
            return list(struct.unpack("256B", f.read(256)))
        else:
            raise Exception(f"bad bytes per pixel: {bytes_per_pixel}")
    else:
        raise Exception("bad image size")

def skip_image(f: BinaryIO, img_size: int, bytes_per_pixel: int) -> None:
    f.seek(img_size*img_size*bytes_per_pixel+1, os.SEEK_CUR)
    
# write an image; image is a list
def write_image_1D(f: BinaryIO, img: list[int], img_size: int, bytes_per_pixel: int) -> None:
    f.write(b'*')
    if img_size == 32:
        if bytes_per_pixel == 1:
            f.write(struct.pack("1024B", *img))
            return
        if bytes_per_pixel == 2:
            f.write(struct.pack("1024H", *img))
            return
    raise Exception('bad params')

# same, image is NxN array
def write_image_2D(f: BinaryIO, img: list[list[int]], img_size: int, bytes_per_pixel: int) -> None:
    f.write(b'*')
    if img_size == 32 and bytes_per_pixel == 2:
        for i in range(32):
            f.write(struct.pack("32H", *img[i]))
        return
    raise Exception('bad params')

# parse a string of the form
# a=b,a=b...a=b.ext
# into a dictionary of a=>b
#
def parse_name(name: str) -> dict[str, str] | None:
    d: dict[str, str] = {}
    n = name.rfind('.')
    if n<0:
        return None
    name = name[0:n]
    x = name.split('.')
    for s in x:
        y = s.split('_')
        if len(y)<2:
            continue
        d[y[0]] = y[1]
    return d

# return the directory name for a run
#
def run_dir_name(obs_name: str, run_type: str) -> str:
    dt = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    dt = dt.replace(microsecond=0)
    dt_str = dt.isoformat()
    return f'obs_{obs_name}.start_{dt_str}Z.runtype_{run_type}.pffd'

def is_pff_dir(name: str) -> bool:
    return name.endswith('.pffd')

def is_pff_file(name: str) -> bool:
    return name.endswith('.pff')

def pff_file_type(name: str) -> str | None:
    if name == 'hk.pff':
        return 'hk'
    n = parse_name(name)
    if n is None or 'dp' not in n:
        return None
    return n['dp']

# return time from parsed JSON header
#
def pkt_header_time(h: dict[str, Any]) -> float:
    # Use precise timing based on Precise-Timing.md 25ms threshold
    # if tv_usec is present.
    if 'tv_usec' in h:
        return wr_to_unix_precise(h['tv_sec'], h['tv_usec'], h['pkt_nsec'])
    return wr_to_unix(h['pkt_tai'], h['pkt_nsec'], h['tv_sec'])

def wr_to_unix_precise(tv_sec: int, tv_usec: int, pkt_nsec: int) -> float:
    """
    Precise timing logic according to Precise-Timing.md (25 ms threshold).
    Compares the NTP fractional second (tv_usec * 1000) against the GPS fractional second (pkt_nsec).
    If they differ by > 25 ms, adjust tv_sec.
    """
    ntp_nsec = tv_usec * 1000
    
    # Is NTP behind GPS by crossing a second boundary?
    # e.g., NTP is 500ms (0.5s), GPS is 30ms. GPS already rolled over.
    if ntp_nsec - pkt_nsec > 25_000_000:
        return (tv_sec + 1) + pkt_nsec / 1e9
    # Is NTP ahead of GPS by crossing a second boundary?
    # e.g., NTP is 30ms (0.03s), GPS is 970ms. NTP already rolled over.
    elif pkt_nsec - ntp_nsec > 25_000_000:
        return (tv_sec - 1) + pkt_nsec / 1e9
    else:
        return tv_sec + pkt_nsec / 1e9

def img_header_time(h: dict[str, Any]) -> float:
    try:
        # this is for img16, img8 and ph1024
        t = pkt_header_time(h['quabo_0'])
    except (KeyError, TypeError):
        # this is for ph256
        t = pkt_header_time(h)
    return t

def img_frame_size(f: BinaryIO, bytes_per_image: int) -> int:
    json_str = read_json(f)
    if json_str:
        json.loads(json_str)
    header_size = f.tell()
    frame_size = header_size + bytes_per_image + 1
    return frame_size

# return info about an image file
#   f points to start of file
#   bytes_per_image: e.g. 1024*2
# returns:
#   frame_size: bytes/frame, including header and image
#   nframes
#   first_t
#   last_t
#
def img_info(f: BinaryIO, bytes_per_image: int) -> tuple[int, int, float, float]:
    json_str = read_json(f)
    if json_str is None:
        return (0, 0, 0.0, 0.0)
    h = json.loads(json_str)
    header_size = f.tell()
    frame_size = header_size + bytes_per_image + 1
    file_size = f.seek(0, os.SEEK_END)
    nframes = int(file_size/frame_size)
    first_t = img_header_time(h)
    i = 1
    while first_t == 0:
        if i >= nframes:
            raise ValueError("All image frames are zero!")
        print('Detected zero frame')
        f.seek(i * frame_size)
        json_str = read_json(f)
        h = json.loads(json_str) if json_str else {}
        # print(h)
        first_t = img_header_time(h)
        # print(first_t)
        i += 1
    f.seek((nframes-1)*frame_size, os.SEEK_SET)
    h_last_str = read_json(f)
    last_t = img_header_time(json.loads(h_last_str)) if h_last_str else 0.0
    return (frame_size, nframes, first_t, last_t)

# return time of given frame
#
def img_frame_time(f: BinaryIO, frame: int, frame_size: int) -> float:
    f.seek(frame*frame_size)
    json_str = read_json(f)
    if json_str:
        s: dict[str, Any] = json.loads(json_str)
        return img_header_time(s)
    return 0.0

# f is a file object, open to the start of an image file
# with integration time frame_time and the given bytes per image.
# Position it (using seek) to a frame whose time is close to t
#
# The file may be missing frames,
# so the frame at the expected position may be after t.
#
def time_seek(f: BinaryIO, frame_time: float, bytes_per_image: int, t: float, verbose: bool = False) -> None:
    first_t: float = 0.0
    nframes: float = float('inf')
    i = 0
    frame_size = 0
    while first_t == 0 and i < nframes:
        (frame_size, nframes_int, first_t, last_t) = img_info(f, bytes_per_image)
        nframes = float(nframes_int)
        i += 1
        f.seek(i * frame_size)

    if t < first_t+frame_time:
        f.seek(0)
        return
    elif t > last_t-frame_time:
        f.seek(int(nframes-1) * frame_size)
        return

    min_t = first_t
    min_f = 0
    max_t = last_t
    max_f = int(nframes)-1

    while True:
        frac = (t-min_t)/(max_t-min_t)
        new_f = min_f + int(frac*(max_f-min_f))
        if new_f <= min_f+1:
            if verbose:
                print(f'new_f {new_f} is close to min_f {min_f}')
            new_f = min_f
            break
        if new_f >= max_f-1:
            if verbose:
                print(f'new_f {new_f} is close to max_f {max_f}')
            break
        new_t = img_frame_time(f, new_f, frame_size)
        if verbose:
            print('new_t', new_t)
        if new_t < t - frame_time:
            min_t = new_t
            min_f = new_f
        elif new_t < t + frame_time:
            if verbose:
                print(f'new_t {new_t:f} is close to t {t:f}')
            f.seek(new_f*frame_size)
            return
        else:
            max_t = new_t
            max_f = new_f
    f.seek(new_f*frame_size)

# Given a WR packet time (TAI) with only 10 bits of sec,
# and a Unix time that's within a few ms,
# return the complete WR time (in Unix time, not TAI)
#
def wr_to_unix(pkt_tai: int, pkt_nsec: int, tv_sec: int, ignore_clock_desync: bool = False) -> float:
    d = (tv_sec - pkt_tai + 37)%1024
    if d == 0:
        return float(tv_sec + pkt_nsec/1e9)
    elif d == 1:
        return float(tv_sec - 1 + pkt_nsec/1e9)
    elif d == 1023:
        return float(tv_sec + 1 + pkt_nsec/1e9)
    else:
        # The WR and DAQ clocks differ by > 1s => out of sync
        # Return 0 if ignore_clock_desync is False. Otherwise, return an approximation to the time.
        if ignore_clock_desync:
            approx_t = tv_sec + pkt_nsec / 1e9
            return float(approx_t)
        else:
            raise Exception(f'WR and Unix times differ by > 1 sec: pkt_tai {pkt_tai} tv_sec {tv_sec} d {d}')


def wr_to_unix_decimal(pkt_tai: int, pkt_nsec: int, tv_sec: int) -> Decimal:
    d_tai = Decimal(str(pkt_tai))
    d_nsec = Decimal(str(pkt_nsec))
    d_tv_sec = Decimal(str(tv_sec))
    nanosec_factor = Decimal(str(1e9))

    d = (d_tv_sec - d_tai + 37)%1024
    if d == 0:
        return d_tv_sec + d_nsec / nanosec_factor
    elif d == 1:
        return d_tv_sec - 1 + d_nsec / nanosec_factor
    elif d == 1023:
        return d_tv_sec + 1 + d_nsec / nanosec_factor
    else:
        return Decimal('0')


def wr_to_unix_numpy(pkt_tai: Any, pkt_nsec: Any, tv_sec: Any) -> Any:
    l_tai = np.longdouble(pkt_tai)
    l_nsec = np.longdouble(pkt_nsec)
    l_tv_sec = np.longdouble(tv_sec)
    d = (l_tv_sec - l_tai + 37)%1024
    if d == 0:
        return l_tai + l_nsec / np.longdouble(1e9)
    elif d == 1:
        return l_tai - 1 + l_nsec / np.longdouble(1e9)
    elif d == 1023:
        return l_tai + 1 + l_nsec / np.longdouble(1e9)
    return np.longdouble(0)


def read_pff_file(path: str) -> Iterator[dict[str, Any]]:
    """Generator that yields JSON headers from a PFF file using PFFSequence."""
    from pydantic import BaseModel

    from control.utils.panoseti_interface import PFFSequence
    
    seq = PFFSequence([path])
    for i in range(len(seq)):
        header, _ = seq.get_frame(i)
        if isinstance(header, BaseModel):
            yield header.model_dump()
        else:
            # It's already a dict
            yield header

