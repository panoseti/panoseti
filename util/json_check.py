#! /usr/bin/env python3

# json.py file.json
# parse and pretty-print the given JSON file.

import sys, json

with open(sys.argv[1]) as f:
    c = json.load(f)
print(json.dumps(c, indent=4))
