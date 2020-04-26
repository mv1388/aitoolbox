#!/usr/bin/env bash

pytest --cov=aitoolbox --cov-report term-missing:skip-covered --cov-report html:html_coverage  tests/

rm -r ./.pytest_cache
sleep 0.5  # In order for the .coverage* files to appear
find  . -name '.coverage*' -exec rm {} \;
