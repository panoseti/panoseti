#! /usr/bin/env python3

# return images from an in-progress run
#
# --ph          return PH images, else image mode
# --nsecs X     return at most 1 image per X sec
# --module N    images from module N
# --bytes_per_pixel N

import time, sys, os

import pff, util

def main(ph, nsecs, module, bytes_per_pixel):
    run = util.daq_get_run_name()
    if not run:
        print('no run')
        return
    dir = 'module_%d/%s'%(module, run)
    file = None
    for f in os.listdir(dir):
        if not pff.is_pff_file(f):
            continue
        finfo = pff.parse_name(f)
        if ph and finfo['dp'] == 'ph':
            file = f
            bytes_per_image = 256*2
            break
        elif finfo['dp'] == 'img16' and bytes_per_pixel == 2:
            file = f
            bytes_per_image = 1024*2
            break
        elif finfo['dp'] == 'img8' and bytes_per_pixel == 1:
            file = f
            bytes_per_image = 1024
            break
    if not file:
        return
    filepath = '%s/%s'%(dir, file)

    # wait for file to be nonempty
    while True:
        if os.path.getsize(filepath):
            break
        time.sleep(1)

    # get file info, e.g. frame size
    print('file: ', filepath)
    f = open(filepath, 'rb')
    (frame_size, nframes, first_t, last_t) = pff.img_info(f, bytes_per_image)

    last_frame = -1
    while True:
        fsize = f.seek(0, os.SEEK_END)
        nframes = int(fsize/frame_size)
        print('fsize: ', fsize, ' nframes: ', nframes, ' last_frame: ', last_frame)
        if nframes > last_frame+1:
            last_frame = nframes-1
            f.seek(last_frame*frame_size, os.SEEK_SET)
            sys.stdout.buffer.write(f.read(frame_size))
        sys.stdout.flush()
        time.sleep(nsecs)

ph = False
nsecs = 1
module = -1
bytes_per_pixel = -1
argv = sys.argv
i = 1
while i<len(argv):
    if argv[i] == '--ph':
        ph = True
    elif argv[i] == '--nsecs':
        i += 1
        nsecs = float(argv[i])
    elif argv[i] == '--module':
        i += 1
        module = int(argv[i])
    elif argv[i] == '--bytes_per_pixel':
        i += 1
        bytes_per_pixel = int(argv[i])
    i += 1
if module < 0:
    print('no module specified')
elif bytes_per_pixel < 0:
    print('no bytes per pixel')
else:
    main(ph, nsecs, module, bytes_per_pixel)
