# functions to read and parse config files

import json

obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'
misc_config_filename = 'misc_config.json'

def get_obs_config():
    with open(obs_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_daq_config():
    with open(daq_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_data_config():
    with open(data_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_misc_config():
    with open(misc_config_filename) as f:
        c = f.read()
    return json.loads(c)

if __name__ == "__main__":
    c = get_obs_config()
    print(c['domes'][1])
