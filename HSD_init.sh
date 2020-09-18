#!/bin/bash
hashpipe -p HSD_hashpipe -I 0 -o BINDHOST="127.0.0.1" -o MAXFILESIZE=50 HSD_net_thread HSD_compute_thread  HSD_output_thread 
