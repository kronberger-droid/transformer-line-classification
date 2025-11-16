#!/bin/bash
# Build script for custom JupyterHub Singularity image on ASC cluster
#
# Usage:
#   For VSC-4: ./build_image.sh vsc4
#   For VSC-5: ./build_image.sh vsc5

set -e  # Exit on error

CLUSTER=${1:-vsc5}  # Default to VSC-5 if not specified

echo "Building Singularity image for $CLUSTER cluster..."
echo "Current directory: $(pwd)"

# Load the appropriate apptainer module based on cluster
if [ "$CLUSTER" == "vsc4" ]; then
    echo "Loading apptainer module for VSC-4..."
    module load --auto apptainer/1.1.9-gcc-12.2.0-vflkcfi
elif [ "$CLUSTER" == "vsc5" ]; then
    echo "Loading apptainer module for VSC-5..."
    module load --auto apptainer/1.1.6-gcc-12.2.0-xxfuqni
else
    echo "Error: Unknown cluster '$CLUSTER'. Please specify 'vsc4' or 'vsc5'"
    exit 1
fi

# Build the image
echo "Building image from jupyter-ml.def..."
apptainer build jupyter-ml.sif jupyter-ml.def

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "Image location: $(pwd)/jupyter-ml.sif"
echo "=========================================="
echo ""
echo "To use this image in JupyterHub:"
echo "1. Select 'Custom image' in the profile options"
echo "2. Enter the full path: $(pwd)/jupyter-ml.sif"
echo ""
