#!/usr/bin/env bash

MODEL_DIR=$1
REMOTE_DIR=$2

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory does not exist!"
    exit 1
fi

cd $MODEL_DIR
if [ ! -d ".git" ]; then
    rm -rf .git
fi

git init

git checkout -b main

git lfs install

git lfs track *.bin

git lfs track *.safetensors

git add .

git commit -m "Initial commit"

git remote add origin $REMOTE_DIR

huggingface-cli lfs-enable-largefiles .

git fetch origin

git push -f --set-upstream origin main
