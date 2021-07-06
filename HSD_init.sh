#!/bin/bash
hashpipe -p HSD_hashpipe -I 0 -o BINDHOST="0.0.0.0" -o MAXFILESIZE=500 -o SAVELOC="/media/panosetigraph/4TB_SSD" HSD_net_thread HSD_compute_thread  HSD_output_thread
