#! /bin/bash

DATA_PATH="$PWD/panoseti_labeling_container_data"

mkdir -p $DATA_PATH

sudo docker rm -f panoseti_data_labeling
sudo docker create \
  --name panoseti_data_labeling \
  -p 8888:8888 \
  -v $DATA_PATH:/home/jovyan/work \
  panoseti_data_labeling:latest \
  start.sh jupyter notebook