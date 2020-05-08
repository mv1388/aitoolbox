#!/usr/bin/env bash

bump_type="patch"
rebuild_pkg=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -t|--type)
    bump_type=$2
    shift 2 # past argument value
    ;;
    -b|--build)
    rebuild_pkg=true
    shift 1 # past argument value
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    exit;
    ;;
esac
done

bumpversion ${bump_type} --config-file .bumpversion.cfg --allow-dirty

if [[ ${rebuild_pkg} == true ]]; then
    mv dist/* dist_old
    ./build_package.sh
fi
