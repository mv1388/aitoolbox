#!/usr/bin/env bash

source activate py36

build_documentation=false
run_unittests=true

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -d|--docu)
    build_documentation=true
    shift 1 # past argument value
    ;;
    -t|--no-test)
    run_unittests=false
    shift 1 # past argument value
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    exit;
    ;;
esac
done


python setup.py sdist bdist_wheel

rm -r aitoolbox.egg-info
rm -r ./.eggs/
rm -r build
git add -A dist/


if [[ "$build_documentation" == true ]]; then
    ./doc_build.sh --clean
fi

# Unittest package and report coverage
if [[ "$run_unittests" == true ]]; then
    ./coverage_test.sh
fi
