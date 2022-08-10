import json
import os, sys
sys.path.insert(1, '/home/panosetigraph/web')
from web_util import *
import pff

def body():
    x = ''
    x += '''
        <h2>Graphical parameter logs</h2>
        <p>
        <a href=http://visigoth.ucolick.org:3000>View</a>
    '''
    x += '''
        <h2>Observing runs</h2>
        <p>
    '''
    x += table_start('table-striped')
    x += table_header(['observatory', 'start', 'run type', 'click to view'])
    runs = []
    for run in os.listdir('/home/panosetigraph/web/data'):
        if not pff.is_pff_dir(run):
            continue
        n = pff.parse_name(run)
        runs.append([n['start'], run])
    runs = sorted(runs, key=lambda x: x[0], reverse=True)
    prev_day = ''
    for run in runs:
        name = run[1]
        n = pff.parse_name(name)
        start = run[0]
        s = start.split('T')
        day = s[0]
        time = s[1]
        if day != prev_day:
            x += table_subheader(day)
            prev_day = day
        x += table_row([
            n['obs'], time, n['runtype'],
            '<a href=run.php?name=%s>View</a>'%name
        ])
    x += table_end();

    return x

#print(body())

def application(environ, start_response):
    start_response('200 OK', [('Content-type','text/html')])
    html = page_head('PanoSETI')+body()+page_tail()
    html = bytes(html, encoding='utf-8')
    return [html]
