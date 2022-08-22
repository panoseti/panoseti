#! /usr/bin/env python3

# process_file.py run
# create video and soft pulse files for the given run

import os, sys
import write_images, pulse, make_dirs

def do_run(run):
    make_dirs.make_dirs(run)
    write_images.do_run(run)
    pulse.do_run(run)

if __name__ == '__main__':
    do_run(sys.argv[1])
