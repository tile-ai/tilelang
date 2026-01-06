#!/usr/bin/env bash

set -eux

rm -rf dist

python -mpip install -U pip
python -mpip install -U build wheel

NO_VERSION_LABEL=1 python -m build --sdist
python -m build --wheel

echo "Wheel built successfully."
