import sys
import json
import time
import requests

sys.path.insert(0, '../util')
dome_control_config = 'dome_control_config.json'


def load_dome_control_config():
    with open(dome_control_config, 'r') as fp:
        return json.load(fp)
