FROM quay.io/jupyter/scipy-notebook:latest

SHELL ["/bin/bash", "-c"]

USER root

# Do a sparse git clone of the cloud-detection-USER branch
RUN git clone --depth 1 \
        --branch cloud-detection-USER \
        --filter=blob:none \
        --no-checkout \
        --sparse \
        https://github.com/panoseti/panoseti.git \
    && cd panoseti \
    && git sparse-checkout set cloud-detection/data_labeling \
    && git checkout cloud-detection-USER -- \
        cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
        cloud-detection/data_labeling/labeling_utils.py \
        cloud-detection/data_labeling/label_session.py \
        cloud-detection/data_labeling/skycam_utils.py \
        cloud-detection/data_labeling/skycam_labels.json \
    && cd .. \
    && mv panoseti .panoseti \
    && ln -s .panoseti/cloud-detection/data_labeling labeling

#WORKDIR /home/jovyan/work/
