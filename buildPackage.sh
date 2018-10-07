#!/usr/bin/env bash

python setup.py test

python setup.py sdist

rm -r AIToolbox.egg-info
git add -A dist/