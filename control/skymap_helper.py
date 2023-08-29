import json, os, time
from datetime import datetime
import redis
from redis.commands.json.path import Path

def add_obs_config(skymap_t, obs_config_file='obs_config.json'):
    with open(obs_config_file) as f:
        obs_config = json.load(f)
    skymap_t['observatory_config']['obs_name'] = obs_config['name']
    domes = obs_config['domes']
    # remove some unnecesssary info
    for dome in domes:
        for m in dome['modules']:
            try:
                m.pop('quabo_version')
                m.pop('wps')
                m.pop('name')
            except:
                pass
    skymap_t['observatory_config']['domes'] = domes

def add_data_config(skymap_t, data_config_file='data_config.json'):
    with open(data_config_file) as f:
        data_config = json.load(f)
    skymap_t['run_type'] = data_config['run_type']
    try:
        skymap_t['data_config']['image'] = data_config['image']
    except:
        pass
    try:
        skymap_t['data_config']['pulse_height']['pe_threshold'] = data_config['pulse_height']['pe_threshold']
    except:
        pass

def add_sw_info(skymap_t, sw='Production', Ver='V0.0.1', sw_info_file='sw_info.json'):
    skymap_t['software_config']['type'] = sw
    # if it's production code, we will get the Ver from sw_info.json
    if(sw == 'Production'):
        with open(sw_info_file) as f:
            sw_info = json.load(f)
        skymap_t['software_config']['version'] = sw_info['commit']
    else:
        skymap_t['software_config']['version'] = Ver

def add_obs_info(skymap_t, obs_com_file='obs_comments.txt'):
    with open(obs_com_file) as f:
        comments = f.readlines()
    skymap_t['observer_info']['names'].append(comments[0].rstrip())
    for c in comments[1:]:
        skymap_t['observer_info']['comments'].append(c.rstrip())
    
def add_format_metadata(skymap_t, entry):
    for k in entry.keys():
        skymap_t[k] = entry[k]

def add_start_time(skymap_t):
    t = time.time()
    dt = datetime.fromtimestamp(t)
    t_str = datetime.strftime(dt, '%Y-%m-%dT%H:%M:%S')
    skymap_t['start_date'] = t_str
    skymap_t['start_unix_t'] = t

def add_stop_time(skymap_t):
    t = time.time()
    dt = datetime.fromtimestamp(t)
    t_str = datetime.strftime(dt, '%Y-%m-%dT%H:%M:%SZ')
    skymap_t['stop_date'] = t_str
    skymap_t['stop_unix_t'] = t
    
def create_empty_entry(template='skymap_format.json'):
    with open(template) as f:
        skymap_t = json.load(f)
    return skymap_t

def write_complete_entry(skymap_t, host='localhost', port=6379):
    client = redis.Redis(host=host, port=port, db=0)
    runs = client.json().get('runs')
    if(runs==None):
        runs = {'runs':[]}
    runs['runs'].append(skymap_t)
    client.json().set('runs', '$', runs)
    return runs