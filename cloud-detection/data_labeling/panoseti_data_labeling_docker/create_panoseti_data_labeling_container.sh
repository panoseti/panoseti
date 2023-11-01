#! /bin/bash

DATA_PATH="$PWD/panoseti_labeling_container_data"

mkdir -p $DATA_PATH

sudo docker run \
  -itd \
  --name panoseti-data-labeling \
  -p 8888:8888 \
  -v $DATA_PATH:/home/jovyan/work \
  panoseti_data_labeling:latest