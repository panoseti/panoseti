#! /usr/bin/env python3

# return images from an in-progress run
#
# --dp          data product (img8/img16/ph)
# --nsecs X     return at most 1 image per X sec
# --module N    images from module N

import time, sys, os

import pff, util

def main(dp, nsecs, module):
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
        if finfo['dp'] == dp:
            file = f
            break
    if not file:
        print('no file of type %s'%dp)
        return
    if dp == 'img8':
        bytes_per_image = 1024
    elif dp == 'img16' or dp == 'ph1024':
        bytes_per_image = 2048
    elif dp == 'ph256':
        bytes_per_image = 512

    filepath = '%s/%s'%(dir, file)

    # wait for file to be nonempty
    while True:
        if os.path.getsize(filepath):
            break
        time.sleep(1)

    # get file info, e.g. frame size
    #print('file: ', filepath)
    f = open(filepath, 'rb')
    (frame_size, nframes, first_t, last_t) = pff.img_info(f, bytes_per_image)

    last_frame = -1
    while True:
        fsize = f.seek(0, os.SEEK_END)
        nframes = int(fsize/frame_size)
        #print('fsize: ', fsize, ' nframes: ', nframes, ' last_frame: ', last_frame)
        if nframes > last_frame+1:
            last_frame = nframes-1
            f.seek(last_frame*frame_size, os.SEEK_SET)
            sys.stdout.buffer.write(f.read(frame_size))
        sys.stdout.flush()
        time.sleep(nsecs)

def do_test():
    while True:
        print('x')
        sys.stdout.flush()
        time.sleep(1)

dp = None
nsecs = 1
module = -1
argv = sys.argv
test = False
i = 1
while i<len(argv):
    if argv[i] == '--dp':
        i += 1
        dp = argv[i]
    elif argv[i] == '--test':
        test = True
    elif argv[i] == '--nsecs':
        i += 1
        nsecs = float(argv[i])
    elif argv[i] == '--module':
        i += 1
        module = int(argv[i])
    i += 1
if test:
    do_test()
elif module < 0:
    print('no module specified')
elif not dp:
    print('no dp specified')
else:
    main(dp, nsecs, module)
