#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

python -m build --wheel -o dist

python setup.py sdist --formats=gztar,zip 

if [ $? -ne 0 ]; then
    echo "Error: Failed to build the wheel."
    exit 1
else
    echo "Wheel built successfully."
fi
