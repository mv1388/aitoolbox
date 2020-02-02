#!/usr/bin/env bash

grep -v '^\r\n*' $1 > $2
