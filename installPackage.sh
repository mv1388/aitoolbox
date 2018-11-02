#!/usr/bin/env bash

pkgversion=$1

source activate /Users/markovidoni/anaconda/envs/Py36

sudo pip install dist/AIToolbox-$pkgversion.tar.gz
