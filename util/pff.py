# functions to parse PFF files,
# and to create and parse PFF dir/file names

import struct
import time, datetime

# returns the string; parse it with json
#
def read_json(f):
    c = f.read(1)
    if c == '':
        return None
    if c != b'{':
        raise Exception('bad type code', c)
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
        else:
            raise Exception("bad bytes per pixel"%bytes_per_pixel)
    elif img_size == 16:
        if bytes_per_pixel == 2:
            return struct.unpack("256H", f.read(512))
        else:
            raise Exception("bad bytes per pixel"%bytes_per_pixel)
    else:
        raise Exception("bad image size"%image_size)

# write an image; image is a list
def write_image_1D(f, img, img_size, bytes_per_pixel):
    f.write(b'*')
    if img_size == 32:
        if bytes_per_pixel == 2:
            f.write(struct.pack("1024H", img))
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
    dp = n['dp']
    if dp == '1':
        return 'img16'
    return dp
