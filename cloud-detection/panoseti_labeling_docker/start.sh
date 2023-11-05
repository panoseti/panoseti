#!/bin/bash

WORK_DIR="/home/jovyan/work"

update_software () {
  # Updates the sparse panoseti repo in the directory specified by parameter #1
  cd "$1"
  git stash 1> /dev/null
  git pull
  git stash pop 1> /dev/null
#  git sparse-checkout set cloud-detection/data_labeling
#  git checkout cloud-detection-USER -- \
#    cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
#    cloud-detection/data_labeling/labeling_utils.py \
#    cloud-detection/data_labeling/label_session.py \
#    cloud-detection/data_labeling/skycam_utils.py \
#    cloud-detection/data_labeling/skycam_labels.json
#  git sparse-checkout reapply
}

# TODO: get rid of spare checkout stuff to simplify automatic code pulling.

if [ ! -d "/home/jovyan/work/panoseti" ]; then
    echo "Setting up labeling software..."

    cd $WORK_DIR
    git clone --branch cloud-detection-USER https://github.com/panoseti/panoseti.git
#    git clone --depth 1 \
#        --branch cloud-detection-USER \
#        --filter=blob:none \
#        --no-checkout \
#        --sparse \
#        https://github.com/panoseti/panoseti.git

    # Only checkout the necessary files for data labeling
#    update_software "$WORK_DIR/panoseti"

    # Create batch dirs
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/batch_data
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/batch_labels

    # Create symbolic link to labeling interface sub-directory
    #ln -s panoseti/cloud-detection/data_labeling labeling

    echo "Done setting up labeling software."
else
    echo "Updating labeling software..."
    update_software "$WORK_DIR/panoseti"
    echo "Done updating labeling software."
fi

jupyter nbclassic --notebook-dir=$WORK_DIR/panoseti/cloud-detection/data_labeling/user_labeling