#!/usr/bin/env bash

# Example call:

# ./installPackage.sh 0.1 --uninstall


pkgversion=$1
uninstallPrevious=$2

source activate py36

if [ $uninstallPrevious == "--uninstall" ]; then
    sudo pip uninstall AIToolbox
fi

sudo pip install dist/AIToolbox-$pkgversion.tar.gz
