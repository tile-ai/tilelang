#!/usr/bin/env bash
set -euxo pipefail

# Build for local architecture
CIBW_BUILD='cp310-*' cibuildwheel . 2>&1 | tee cibuildwheel.log
