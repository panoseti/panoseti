"""
Scripts for generating redis-json for sky map.
"""

import json, os, time
from datetime import datetime
import redis
from redis.commands.json.path import Path

# Add obs config to the skymap template.
# The obs config is from obs_config.json by default.
#
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

# Add data config to the skymap template.
# The data config is from data_config.json by default.
#
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

# Add software version info to the skymap template.
# The software version is from sw_info.json by default. 
# 
def add_sw_info(skymap_t, sw='Production', Ver='V0.0.1', sw_info_file='sw_info.json'):
    skymap_t['software_config']['type'] = sw
    # if it's production code, we will get the Ver from sw_info.json
    if(sw == 'Production'):
        with open(sw_info_file) as f:
            sw_info = json.load(f)
        skymap_t['software_config']['version'] = sw_info['commit']
    else:
        skymap_t['software_config']['version'] = Ver

# Add obs comments to the skymap template.
# The comments are from obs_comments.txt by default.
#
def add_obs_info(skymap_t, obs_com_file='obs_comments.txt'):
    with open(obs_com_file) as f:
        comments = f.readlines()
    skymap_t['observer_info']['names'].append(comments[0].rstrip())
    for c in comments[1:]:
        skymap_t['observer_info']['comments'].append(c.rstrip())
    
# Add start time to the skymap template.
#
def add_start_time(skymap_t):
    t = time.time()
    dt = datetime.fromtimestamp(t)
    t_str = datetime.strftime(dt, '%Y-%m-%dT%H:%M:%S')
    skymap_t['start_date'] = t_str
    skymap_t['start_unix_t'] = t

# Add stop time to the skymap template.
#
def add_stop_time(skymap_t):
    t = time.time()
    dt = datetime.fromtimestamp(t)
    t_str = datetime.strftime(dt, '%Y-%m-%dT%H:%M:%SZ')
    skymap_t['stop_date'] = t_str
    skymap_t['stop_unix_t'] = t

# Create a skymap template, which is from skymap_format.json.
#    
def create_empty_entry(template='skymap_format.json'):
    with open(template) as f:
        skymap_t = json.load(f)
    return skymap_t

# Write the skymap data into redis server.
#
def write_complete_entry(skymap_t, host='localhost', port=6379):
    client = redis.Redis(host=host, port=port, db=0)
    runs = client.json().get('runs')
    if(runs==None):
        runs = {'runs':[]}
    runs['runs'].append(skymap_t)
    client.json().set('runs', '$', runs)
    return runs

def start_skymap_info_gen(skymap_info_file='skymap_info.json'):
    skymap_t = create_empty_entry()
    add_obs_config(skymap_t)
    add_data_config(skymap_t)
    add_sw_info(skymap_t)
    add_obs_info(skymap_t)
    add_start_time(skymap_t)
    # dump the json info
    json_object = json.dumps(skymap_t, indent=4)
    # write the data to skymap_info.json.
    # We will read this json file, and add stop time after the observation.
    with open(skymap_info_file,'w') as f:
        f.write(json_object)

def stop_skymap_info_gen(skymap_info_file='skymap_info.json'):
    with open(skymap_info_file) as f:
        skymap_t = json.load(f)
    add_stop_time(skymap_t)
    # write the data back to the json file
    json_object = json.dumps(skymap_t, indent=4)
    with open(skymap_info_file,'w') as f:
        f.write(json_object)
    write_complete_entry(skymap_t)