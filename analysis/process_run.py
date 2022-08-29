#! /usr/bin/env python3

# process_file.py run
# create video and soft pulse files for the given run

import os, sys
import write_images, pulse, analysis_util

def do_run(run):
    analysis_util.make_dirs(run)
    write_images.do_run(run)
    pulse.do_run(run)
    summary = {}
    summary['images'] = '.mp4 of first 1000 frames'
    summary['image_pulse'] = '6 pixels; nlevels 16, thresh 3'
    analysis_util.write_summary(run, summary)

if __name__ == '__main__':
    do_run(sys.argv[1])
