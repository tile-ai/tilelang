#!/usr/bin/env bash

nvidia-smi --query-gpu=memory.used --format=csv -lms 500
