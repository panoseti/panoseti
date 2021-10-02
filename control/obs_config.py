# function to parse the master config file

import json

obs_config_filename = 'obs_config.json'

def get_config():
    with open(obs_config_filename) as f:
        c = f.read()
    return json.loads(c)
