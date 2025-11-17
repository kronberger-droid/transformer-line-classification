#!/bin/bash
# Sync code to cluster (excludes data and outputs)

# CONFIGURE THESE:
CLUSTER_USER="your_username"
CLUSTER_HOST="cluster.address"
CLUSTER_PATH="/path/to/your/project/"

# Sync to cluster
rsync -avz --progress \
  --exclude='data/*.npz' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='*.pt' \
  --exclude='.git/lfs/objects/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  ./ \
  ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}

echo "Sync complete!"
