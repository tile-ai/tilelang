#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

./maint/scripts/docker_local_distribute.sh 2>&1 | tee docker_local_distribute.log

./maint/scripts/docker_pypi_distribute.sh 2>&1 | tee docker_pypi_distribute.log
