#!/usr/bin/env bash

source activate py36

python setup.py test

python setup.py sdist

rm -r aitoolbox.egg-info
rm -r ./.eggs/
git add -A dist/