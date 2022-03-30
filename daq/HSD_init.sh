#!/bin/bash
hashpipe -p ./hashpipe.so -I 0 -o BINDHOST="0.0.0.0" -o MAXFILESIZE=500 -o RUNDIR="" -o CONFIG="./module.config" net_thread compute_thread  output_thread
