#! /bin/bash

sudo docker pull quay.io/jupyter/scipy-notebook:latest
sudo docker rmi panoseti_data_labeling:latest
sudo docker build -t panoseti_data_labeling:latest .


