#!/usr/bin/env bash

# Example call:

# ./install_package.sh 1.8.0 --uninstall


pkgversion=$1
uninstallPrevious=$2

source activate py39

if [ $uninstallPrevious == "--uninstall" ]; then
    sudo pip uninstall aitoolbox
fi

sudo pip install dist/aitoolbox-$pkgversion.tar.gz
