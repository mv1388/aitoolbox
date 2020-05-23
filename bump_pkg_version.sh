#!/usr/bin/env bash

# Using the bump2version package to search for version strings and bump them
#
#   https://github.com/c4urself/bump2version


bump_type="patch"
rebuild_pkg=true
gpu_test=true
fast_gpu_test=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -t|--type)
    bump_type=$2
    shift 2 # past argument value
    ;;
    -b|--no-build)
    rebuild_pkg=false
    shift 1 # past argument value
    ;;
    -g|--no-gpu-test)
    gpu_test=false
    shift 1 # past argument value
    ;;
    -f|--fast-gpu-test)
    fast_gpu_test=true
    shift 1 # past argument value
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    exit;
    ;;
esac
done

bumpversion ${bump_type} --config-file .bumpversion.cfg

if [[ ${rebuild_pkg} == true ]]; then
    mv dist/* dist_old
    ./build_package.sh --docu
fi

if [[ ${gpu_test} == true ]]; then
    cd bin/AWS
    echo "Submitting GPU tests"

    if [[ ${fast_gpu_test} == false ]]; then
        echo "Submitting Single GPU tests"
        ./test_core_pytorch_compare.sh --no-ssh
        echo "Submitting Multi GPU tests"
        ./test_core_pytorch_compare.sh --multi-gpu --instance-type p2.8xlarge --no-ssh
    else
        echo "Submitting Single GPU tests"
        ./test_core_pytorch_compare.sh --instance-type p3.2xlarge --no-ssh
        echo "Submitting Multi GPU tests"
        ./test_core_pytorch_compare.sh --multi-gpu --instance-type p3.8xlarge --no-ssh
    fi

    echo "GPU test submissions done!"
    echo "Wait for these tests to finish successfully before creating the new package release on GitHub and deployment!"
fi
