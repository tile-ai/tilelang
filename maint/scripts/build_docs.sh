#!/usr/bin/env bash

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --no-user
python -m pip install -r docs/requirements.txt --no-user

cd docs
make html

cp CNAME _build/html/
