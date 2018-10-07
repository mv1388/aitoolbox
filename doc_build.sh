#!/usr/bin/env bash

sphinx-apidoc -f -o docs/source AIToolbox/

cd docs
make html
cd ..
