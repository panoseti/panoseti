#!/bin/bash

WORK_DIR="/home/jovyan/work"

update_software () {
  # Updates the sparse panoseti repo in the directory specified by parameter #1
  cd "$1"
  git stash 1> /dev/null
  git pull
  git stash pop 1> /dev/null
}

if [ ! -d "/home/jovyan/work/panoseti" ]; then
    echo "Setting up labeling software..."

    cd $WORK_DIR
    git clone --branch cloud-detection-USER https://github.com/panoseti/panoseti.git

    # Create batch dirs
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/user_labeling/batch_data
    mkdir $WORK_DIR/panoseti/cloud-detection/data_labeling/user_labeling/batch_labels
    echo "Done setting up labeling software."
else
    echo "Updating labeling software..."
    update_software "$WORK_DIR/panoseti"
    echo "Done updating labeling software."
fi

jupyter nbclassic --notebook-dir=$WORK_DIR/panoseti/cloud-detection/data_labeling/user_labeling