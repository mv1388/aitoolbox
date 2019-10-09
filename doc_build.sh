#!/usr/bin/env bash

sphinx-apidoc -f -o docs/source aitoolbox/

cd docs
make html
cd ..
