#!/usr/bin/env bash

pytest --cov=aitoolbox --cov-report term-missing:skip-covered --cov-report html:html_coverage  tests/

rm -r ./.pytest_cache
find  . -name '.coverage*' -exec rm {} \;
