#!/usr/bin/env bash

#grep -v '^\r\n*' "$1" > "$2"

sed -n -e '/              STARTING THE TRAINING JOB               /, $p' "$1" | grep -v '%|.*|' > "$2"
