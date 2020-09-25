#!/bin/bash
hashpipe -p HSD_hashpipe -I 0 -o BINDHOST="0.0.0.0" -o MAXFILESIZE=50 HSD_net_thread HSD_compute_thread  HSD_output_thread 
