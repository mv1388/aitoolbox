#!/usr/bin/env bash

source activate py36

sphinx-build -b html docs/source docs/build
