#!/usr/bin/env bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

nvidia-smi --query-gpu=memory.used --format=csv -lms 500
