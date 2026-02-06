#!/usr/bin/env bash
set -euxo pipefail

# Build for local architecture
CIBW_BUILD='cp39-*' cibuildwheel . 2>&1 | tee cibuildwheel.log
