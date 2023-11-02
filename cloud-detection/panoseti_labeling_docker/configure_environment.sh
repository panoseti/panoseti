#!/bin/bash

WORK_DIR="/home/jovyan/work"
if [ ! -d "/home/jovyan/work/.panoseti" ]; then
    # Install dependencies
    pip install -r ~/work/requirements.txt
    export PYTHONPATH=PYTHONPATH:/home/jovyan/work

    echo "Setting up labeling software..."

    # Do a sparse git clone of the cloud-detection-USER branch
    cd $WORK_DIR
    git clone --depth 1 \
        --branch cloud-detection-USER \
        --filter=blob:none \
        --no-checkout \
        --sparse \
        https://github.com/panoseti/panoseti.git

    # Only checkout the necessary files for data labeling
    cd $WORK_DIR/panoseti
    git sparse-checkout set cloud-detection/data_labeling
    git checkout cloud-detection-USER -- \
        cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
        cloud-detection/data_labeling/labeling_utils.py \
        cloud-detection/data_labeling/label_session.py \
        cloud-detection/data_labeling/skycam_utils.py \
        cloud-detection/data_labeling/skycam_labels.json

    # Create batch data dir
    mkdir /home/jovyan/work/panoseti/cloud-detection/data_labeling/batch_data

    # Make panoseti volume hidden for simplicity
    cd $WORK_DIR
    mv $WORK_DIR/panoseti $WORK_DIR/.panoseti

    # Create symbolic link to labeling interface sub-directory
    ln -s .panoseti/cloud-detection/data_labeling labeling
    echo "Done setting up labeling software."
else
    cd $WORK_DIR/.panoseti
    echo "Checking for labeling software updates..."
    if git fetch | grep -q '*'; then
      echo "Updating labeling software..."
      git stash 1> /dev/null
      git pull
      git stash pop 1> /dev/null
      echo "Done updating labeling software."
    else
      echo "No updates found."
    fi
fi