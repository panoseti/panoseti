FROM quay.io/jupyter/scipy-notebook:latest

SHELL ["/bin/bash", "-c"]

USER jovyan

# Do a sparse git clone of the cloud-detection-USER branch
RUN cd /home/jovyan/ \
    && git clone --depth 1 \
        --branch cloud-detection-USER \
        --filter=blob:none \
        --no-checkout \
        --sparse \
        https://github.com/panoseti/panoseti.git \
    && cd panoseti \
    && git sparse-checkout set cloud-detection/data_labeling \
    # Only checkout the necessary files for data labeling
    && git checkout cloud-detection-USER -- \
        cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
        cloud-detection/data_labeling/labeling_utils.py \
        cloud-detection/data_labeling/label_session.py \
        cloud-detection/data_labeling/skycam_utils.py \
        cloud-detection/data_labeling/skycam_labels.json \
        cloud-detection/data_labeling/start-notebook.py \
    # Use customized start-notebook.py script
    && mv cloud-detection/data_labeling/start-notebook.py /usr/local/bin/start-notebook.py \
    # Create symbolic link to labeling interface sub-directory
    && cd .. \
    && mv panoseti .panoseti \
    && ln -s .panoseti/cloud-detection/data_labeling labeling \
    && mkdir -p labeling/batch_data \



WORKDIR /home/jovyan/labeling
