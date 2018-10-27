#!/usr/bin/env bash

pkgversion=$1

source activate Py36

sudo pip install dist/AIToolbox-$pkgversion.tar.gz
