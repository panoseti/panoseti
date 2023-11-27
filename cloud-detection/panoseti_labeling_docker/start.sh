#!/usr/bin/env bash
set -x
pid=0

WORK_DIR="/home/jovyan/work"

# Updates panoseti repo in the directory specified by parameter #1
update_software () {
  cd "$1"
  git stash 1> /dev/null
  git pull
  git stash pop 1> /dev/null
}

# SIGTERM-handler
term_handler() {
  if [ $pid -ne 0 ]; then
    kill -SIGTERM "$pid"
    wait "$pid"
  fi
  exit 143; # 128 + 15 -- SIGTERM
}

if [ ! -d "/home/jovyan/work/panoseti" ]; then
    echo "Setting up labeling software..."

    # Pull panoseti repo
    cd $WORK_DIR
    git clone --branch cloud-detection-USER https://github.com/panoseti/panoseti.git

    # Create batch dirs
    mkdir $WORK_DIR/panoseti/cloud-detection/dataset_construction/user_labeling/batch_data
    mkdir $WORK_DIR/panoseti/cloud-detection/dataset_construction/user_labeling/batch_labels
    echo "Done setting up labeling software."
else
    echo "Updating labeling software..."
    update_software "$WORK_DIR/panoseti"
    echo "Done updating labeling software."
fi

# Handle SIGTERM signal upon container exit
trap 'kill ${!}; term_handler' SIGTERM

# Run labeling notebook
# NOTE: for simplicity we've disabled security for the notebook, so only run this on your local machine...
jupyter nbclassic \
  --notebook-dir=$WORK_DIR/panoseti/cloud-detection/dataset_construction/user_labeling \
  --port=16113 \
  --no-browser \
  --NotebookApp.password='' \
  --NotebookApp.token='' \
  &
pid="$!"

# Wait until sigterm
while true
do
  tail -f /dev/null & wait ${!}
done