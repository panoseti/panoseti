# utility functions for analysis
#
# see https://github.com/panoseti/panoseti/wiki/Analysis-framework

import os, shutil, json, datetime

# analysis types
#
ANALYSIS_TYPE_IMAGE_PULSE = 'image_pulse'

ANALYSIS_ROOT = 'analysis'

# write a JSON file saying when analysis was done and with what params
#
def write_summary(analysis_dir, params):
    summary = {}
    summary['when'] = datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
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
def make_analysis_dir(analysis_type, run=None):
    make_dir(ANALYSIS_ROOT)
    if run:
        run_dir = make_dir('%s/%s'%(ANALYSIS_ROOT, run))
        type_dir = make_dir('%s/%s'%(run_dir, analysis_type))
    else:
        type_dir = make_dir('%s/%s'%(ANALYSIS_ROOT, analysis_type))
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    return make_dir('%s/%s'%(type_dir, now))
