# parse PFF files and dir/file names

import struct
import time, datetime

# returns the string; parse it with json
#
def read_json(f):
    c = f.read(1)
    if c != b'{':
        print('bad type code')
        return Null
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

# returns the image as a list of 1024 numbers
# see https://docs.python.org/3/library/struct.html
#
def read_image_16(f):
    c = f.read(1)
    if c != b'*':
        print('bad type code')
        return Null
    return struct.unpack("1024H", f.read(2048))

# write an image; image is 1024
def write_image_16_1(f, img):
    f.write(b'*')
    f.write(struct.pack("1024H", img))

# same, image is 32x32
def write_image_16_2(f, img):
    f.write(b'*')
    for i in range(32):
        f.write(struct.pack("32H", *img[i]))

# parse a string of the form
# a=b,a=b...a=b.ext
# into a dictionary of a=>b
#
def parse_name(name):
    d = {}
    n = name.rfind('.')
    if n<0:
        return Null
    name = name[0:n]
    x = name.split('.')
    for s in x:
        y = s.split('_')
        d[y[0]] = y[1]
    return d

# return the directory name for a run
#
def run_dir_name(obs_name, run_type):
    dt = datetime.datetime.utcnow()
    dt = dt.replace(microsecond=0)
    dt_str = dt.isoformat()
    return 'obs_%s.start_%sZ.runtype_%s'%(obs_name, dt_str, run_type)
