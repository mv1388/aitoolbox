#!/usr/bin/env bash

source activate py36

#sphinx-apidoc -f --separate --module-first -t docs/source/_templates -o docs/source/api aitoolbox

sphinx-build -b html docs/source docs/build
