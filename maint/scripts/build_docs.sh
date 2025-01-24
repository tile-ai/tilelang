#!/bin/bash

cd docs

pip install -r requirements.txt

make html

cp CNAME _build/html/


