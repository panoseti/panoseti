WORK_DIR="/home/jovyan/work"
if [ ! -d "/home/jovyan/work/.panoseti" ]; then
    cd $WORK_DIR
    # Do a sparse git clone of the cloud-detection-USER branch
    git clone --depth 1 \
        --branch cloud-detection-USER \
        --filter=blob:none \
        --no-checkout \
        --sparse \
        https://github.com/panoseti/panoseti.git
    cd $WORK_DIR/panoseti
    git sparse-checkout set cloud-detection/data_labeling

    # Only checkout the necessary files for data labeling
    git checkout cloud-detection-USER -- \
        cloud-detection/data_labeling/PANOSETI\ Data\ Labeling\ Interface.ipynb \
        cloud-detection/data_labeling/labeling_utils.py \
        cloud-detection/data_labeling/label_session.py \
        cloud-detection/data_labeling/skycam_utils.py \
        cloud-detection/data_labeling/skycam_labels.json
    mkdir /home/jovyan/work/panoseti/cloud-detection/data_labeling/batch_data

    # Make panoseti volume hidden
    cd $WORK_DIR
    mv $WORK_DIR/panoseti $WORK_DIR/.panoseti

    # Create symbolic link to labeling interface sub-directory
    ln -s .panoseti/cloud-detection/data_labeling labeling
    echo "\033[32mAll files downloaded\033[0m"
else
    cd $WORK_DIR/.panoseti
    git pull
    echo "The labeling directory already exists."
fi
