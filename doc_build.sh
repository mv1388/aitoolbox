#!/usr/bin/env bash

sphinx-apidoc -f -o docs/source AIToolbox/

cd docs
sphinx-apidoc -f -o source ../
make html
cd ..
