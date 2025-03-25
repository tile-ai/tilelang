#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

if [ -d build ]; then
    rm -r build
fi

PYPI_BUILD=TRUE python setup.py bdist_wheel --plat-name=manylinux1_x86_64
