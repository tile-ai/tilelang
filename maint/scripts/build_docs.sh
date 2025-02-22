#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cd docs

pip install -r requirements.txt

make html

cp CNAME _build/html/
