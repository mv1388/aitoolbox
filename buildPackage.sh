#!/usr/bin/env bash

source activate Py36

python setup.py test

python setup.py sdist

rm -r AIToolbox.egg-info
rm -r ./.eggs/
git add -A dist/