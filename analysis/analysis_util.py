# utility functions for analysis

import os, shutil, json, datetime

# write a JSON file saying when analysis was done and with what params
#
def write_summary(run, summary):
    summary['when'] = datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    with open('derived/%s/summary.json'%run, 'w') as f:
        f.write(json.dumps(summary, indent=4))

# create run and file directories as needed for derived files
#
def make_dirs(run=None, f=None):
    if not os.path.exists('derived'):
        os.mkdir('derived')
        shutil.chown('derived', group='panosetigraph')
        os.chmod('derived', 0o775)
    if run:
        if not os.path.exists('derived/%s'%run):
            os.mkdir('derived/%s'%run)
            shutil.chown('derived/%s'%run, group='panosetigraph')
            os.chmod('derived/%s'%run, 0o775)
    if f:
        if not run:
            raise Exception('must specify run')
        if not os.path.exists('derived/%s/%s'%(run, f)):
            os.mkdir('derived/%s/%s'%(run, f))
            shutil.chown('derived/%s/%s'%(run, f), group='panosetigraph')
            os.chmod('derived/%s/%s'%(run, f), 0o775)

