# parsing a PFF file in Python

from struct import *

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
    return unpack("1024H", f.read(2048))

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
    x = name.split(',')
    for s in x:
        y = s.split('=')
        d[y[0]] = y[1]
    return d
