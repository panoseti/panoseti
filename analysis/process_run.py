#! /usr/bin/env python3

# process_file.py run
# create video and soft pulse files for the given run

DEPRECATED; REPURPOSE FOR IMAGE ONLY

import os, sys
import write_images, pulse, analysis_util

def do_run(run):
    analysis_util.make_dirs(run)
    write_images.do_run(run)
    pulse.do_run(run)
    summary = {}
    summary['images'] = '.mp4 of first 1000 frames'
    analysis_util.write_summary(run, summary)

if __name__ == '__main__':
    do_run(sys.argv[1])
