# functions to parse PFF files,
# and to create and parse PFF dir/file names

import struct, os, time, datetime, json

# returns the string (doesn't parse it)
#
def read_json(f):
    c = f.read(1)
    if c == '':
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
    return s;

# returns the image as a list of N numbers
# see https://docs.python.org/3/library/struct.html
#
def read_image(f, img_size, bytes_per_pixel):
    c = f.read(1)
    if c == '':
        return None
    if c != b'*':
        raise Exception('bad type code')
    if img_size == 32:
        if bytes_per_pixel == 2:
            return struct.unpack("1024H", f.read(2048))
        elif bytes_per_pixel == 1:
            return struct.unpack("1024B", f.read(1024))
        else:
            raise Exception("bad bytes per pixel: %d"%bytes_per_pixel)
    elif img_size == 16:
        if bytes_per_pixel == 2:
            return struct.unpack("256H", f.read(512))
        elif bytes_per_pixel == 2:
            return struct.unpack("256B", f.read(256))
        else:
            raise Exception("bad bytes per pixel: %d"%bytes_per_pixel)
    else:
        raise Exception("bad image size"%img_size)

def skip_image(f, img_size, bytes_per_pixel):
    f.seek(img_size*img_size*bytes_per_pixel+1, os.SEEK_CUR)
    
# write an image; image is a list
def write_image_1D(f, img, img_size, bytes_per_pixel):
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
def write_image_2D(f, img, img_size, bytes_per_pixel):
    f.write(b'*')
    if img_size == 32:
        if bytes_per_pixel == 2:
            for i in range(32):
                f.write(struct.pack("32H", *img[i]))
            return
    raise Exception('bad params')

# parse a string of the form
# a=b,a=b...a=b.ext
# into a dictionary of a=>b
#
def parse_name(name):
    d = {}
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
def run_dir_name(obs_name, run_type):
    dt = datetime.datetime.utcnow()
    dt = dt.replace(microsecond=0)
    dt_str = dt.isoformat()
    return 'obs_%s.start_%sZ.runtype_%s.pffd'%(obs_name, dt_str, run_type)

def is_pff_dir(name):
    return name.endswith('.pffd')

def is_pff_file(name):
    return name.endswith('.pff')

def pff_file_type(name):
    if name == 'hk.pff':
        return 'hk'
    n = parse_name(name)
    if 'dp' not in n.keys():
        return None
    return n['dp']

# return time from parsed JSON header
#
def pkt_header_time(h):
    return wr_to_unix(h['pkt_tai'], h['pkt_nsec'], h['tv_sec'])

def img_header_time(h):
    return pkt_header_time(h['quabo_0'])

# return info about an image file
#   f points to start of file
#   bytes_per_image: e.g. 1024*2
# returns:
#   frame_size: bytes/frame, including header and image
#   nframes
#   first_t
#   last_t
#
def img_info(f, bytes_per_image):
    h = json.loads(read_json(f))
    header_size = f.tell()
    frame_size = header_size + bytes_per_image + 1
    file_size = f.seek(0, os.SEEK_END)
    nframes = int(file_size/frame_size)
    first_t = img_header_time(h)
    f.seek((nframes-1)*frame_size, os.SEEK_SET)
    h = json.loads(read_json(f))
    last_t = img_header_time(h)
    return [frame_size, nframes, first_t, last_t]

# return time of given frame
#
def img_frame_time(f, frame, frame_size):
    f.seek(frame*frame_size)
    s = json.loads(read_json(f))
    return img_header_time(s)

# f is a file object, open to the start of an image file
# with integration time frame_time and the given bytes per image.
# Position it (using seek) to a frame whose time is close to t
#
# The file may be missing frames,
# so the frame at the expected position may be after t.
#
def time_seek(f, frame_time, bytes_per_image, t, verbose=False):
    (frame_size, nframes, first_t, last_t) = img_info(f, bytes_per_image)

    if t < first_t+frame_time:
        f.seek(0)
        return
    elif t > last_t-frame_time:
        f.seek(int(nframes-1) * frame_size)
        return

    min_t = first_t
    min_f = 0
    max_t = last_t
    max_f = nframes-1

    while True:
        frac = (t-min_t)/(max_t-min_t)
        new_f = min_f + int(frac*(max_f-min_f))
        if new_f <= min_f+1:
            if verbose:
                print('new_f %d is close to min_f %d'%(new_f, min_f))
            new_f = min_f
            break
        if new_f >= max_f-1:
            if verbose:
                print('new_f %d is close to max_f %d'%(new_f, max_f))
            break
        new_t = img_frame_time(f, new_f, frame_size)
        if verbose:
            print('new_t', new_t)
        if new_t < t - frame_time:
            min_t = new_t
            min_f = new_f
        elif new_t < t + frame_time:
            if verbose:
                print('new_t %f is close to t %f'%(new_t, t))
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
def wr_to_unix(pkt_tai, pkt_nsec, tv_sec):
    d = (tv_sec - pkt_tai + 37)%1024
    if d == 0:
        return tv_sec + pkt_nsec/1e9
    elif d == 1:
        return tv_sec - 1 + pkt_nsec/1e9
    elif d == 1023:
        return tv_sec + 1 + pkt_nsec/1e9
    else:
        return 0
        #raise Exception('WR and Unix times differ by > 1 sec: pkt_tai %d tv_sec %d d %d'%(pkt_tai, tv_sec, d))
