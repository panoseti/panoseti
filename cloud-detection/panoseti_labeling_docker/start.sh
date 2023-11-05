#!/bin/bash

WORK_DIR="/home/jovyan/work"

update_software () {
  # Updates the sparse panoseti repo in the directory specified by parameter #1
  cd "$1"
  git sparse-checkout set cloud-detection/data_labeling
  git checkout cloud-detection-USER -- \
    cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
    cloud-detection/data_labeling/labeling_utils.py \
    cloud-detection/data_labeling/label_session.py \
    cloud-detection/data_labeling/skycam_utils.py \
    cloud-detection/data_labeling/skycam_labels.json
  git sparse-checkout reapply
}

if [ ! -d "/home/jovyan/work/panoseti" ]; then
    #export PYTHONPATH=PYTHONPATH:$WORK_DIR

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
    update_software "$WORK_DIR/panoseti"

    # Create batch dirs
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/batch_data
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/batch_labels

    # Create symbolic link to labeling interface sub-directory
    #ln -s panoseti/cloud-detection/data_labeling labeling

    echo "Done setting up labeling software."
else
    cd $WORK_DIR/panoseti
    echo "Checking for labeling software updates..."
    if [ ! -z "$(git fetch | grep ".*")" ]; then
      echo "Updating labeling software..."
      #git stash 1> /dev/null
      update_software "$WORK_DIR/panoseti"
      #git stash pop 1> /dev/null
      echo "Done updating labeling software."
    else
      echo "No updates found."
    fi
fi

jupyter nbclassic \
  --notebook-dir=$WORK_DIR/panoseti/cloud-detection/data_labeling