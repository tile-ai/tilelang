#!/usr/bin/env bash
set -euxo pipefail

if docker buildx version >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    docker run --rm --privileged tonistiigi/binfmt --install amd64,arm64 >/dev/null 2>&1 || true
  fi

  if ! docker buildx inspect multi >/dev/null 2>&1; then
    docker buildx create --name multi --driver docker-container --use >/dev/null 2>&1 || true
  else
    docker buildx use multi >/dev/null 2>&1 || true
  fi
  docker buildx inspect --bootstrap >/dev/null 2>&1 || true

  export CIBW_ARCHS='x86_64 aarch64'
fi

NO_VERSION_LABEL=ON CIBW_BUILD='cp39-*' cibuildwheel . 2>&1 | tee cibuildwheel.log
