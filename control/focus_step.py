#! /usr/bin/env python3

# step the focus of a quabo
#
# focus_step.py ip_addr start stop step dt
#
# ip_addr   IP address of quabo
# start     starting focus value (1..50000)
# end       ending value
# step      step size
# dt        sec per step

import sys, time, quabo_driver

def main(ip_addr, start, end, step, dt):
    q = quabo_driver.QUABO(ip_addr)
    val = start
    while val < end:
        q.focus(val)
        val += step
        time.sleep(dt)

main(
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    float(sys.argv[5])
)
