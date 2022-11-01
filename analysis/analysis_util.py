# utility functions for analysis
#
# see https://github.com/panoseti/panoseti/wiki/Analysis-framework

import os, sys, shutil, json, datetime
sys.path.append('../util')
import config_file

# analysis types
#
ANALYSIS_TYPE_IMAGE_PULSE = 'img_pulse'
ANALYSIS_TYPE_VISUAL = 'visual'

ANALYSIS_ROOT = 'analysis'

# write a JSON file saying when analysis was done and with what params
#
def write_summary(analysis_dir, params, username):
    summary = {}
    summary['when'] = datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    summary['username'] = username
    summary['params'] = params
    with open('%s/summary.json'%analysis_dir, 'w') as f:
        f.write(json.dumps(summary, indent=4))

# if dir doesn't exist, create it and set ownership and permissions
# so that both panosetigraph and www-data can r/w it
#
def make_dir(path):
    if os.path.exists(path):
        return path
    os.mkdir(path)
    shutil.chown(path, group='panosetigraph')
    os.chmod(path, 0o775)
    return path

# create an analysis run directory; return path
#
def make_analysis_dir(analysis_type, vol=None, run=None):
    make_dir(ANALYSIS_ROOT)
    if run:
        run_dir = make_dir('%s/%s/%s'%(vol, ANALYSIS_ROOT, run))
        type_dir = make_dir('%s/%s'%(run_dir, analysis_type))
    else:
        type_dir = make_dir('%s/%s'%(ANALYSIS_ROOT, analysis_type))
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    return make_dir('%s/%s'%(type_dir, now))

# convert nsecs to nframes, based on frame rate
#
def img_seconds_to_frames(vol, run, seconds):
    data_config = config_file.get_data_config('%s/data/%s'%(vol, run))
    x = float(data_config['image']['integration_time_usec'])
    return seconds*1.e6/x

