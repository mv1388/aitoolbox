#!/usr/bin/env bash

source activate py39

clean_build=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -c|--clean)
    clean_build=true
    shift 1 # past argument value
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    exit;
    ;;
esac
done


#sphinx-apidoc -f --separate --module-first -t docs/source/_templates -o docs/source/api aitoolbox

if [ "$clean_build" == false ]; then
    sphinx-build -b html docs/source docs/build
else
    sphinx-build -Ea -b html docs/source docs/build
fi

