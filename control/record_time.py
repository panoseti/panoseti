#! /usr/bin/env python

# record_time.py dir
#
# write one line per second to dir/time
#
# Placeholder for hashpipe program while debugging DAQ code

import sys, time, datetime

dir = sys.argv[1]

with open('%s/time'%(dir), 'w') as f:
    while True:
        t = int(time.time())
        dt = datetime.datetime.fromtimestamp(t)
        dt_str = dt.isoformat()
        f.write(dt_str)
        f.write('\n')
        f.flush()
        time.sleep(1)
