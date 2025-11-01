#!/usr/bin/env bash
set -euxo pipefail

# Build for local architecture
CIBW_BUILD='cp38-*' cibuildwheel .
