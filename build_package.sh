#!/usr/bin/env bash

source activate py36

build_documentation=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -d|--docu)
    build_documentation=true
    shift 1 # past argument value
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    ;;
esac
done


python setup.py test

python setup.py sdist

rm -r aitoolbox.egg-info
rm -r ./.eggs/
git add -A dist/


if [ "$build_documentation" == true ]; then
    ./doc_build.sh --clean
fi
