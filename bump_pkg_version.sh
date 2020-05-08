#!/usr/bin/env bash

current_version=1.0.2
bump_type="patch"
rebuild_pkg=true

bump2version --current-version ${current_version} ${bump_type} setup.py
bump2version --current-version ${current_version} ${bump_type} install_package.sh
bump2version --current-version ${current_version} ${bump_type} bin/AWS/create_instance.sh
bump2version --current-version ${current_version} ${bump_type} bin/AWS/prepare_instance.sh
bump2version --current-version ${current_version} ${bump_type} bin/AWS/submit_job.sh
bump2version --current-version ${current_version} ${bump_type} bin/AWS/update_package_on_AWS.sh


if [[ ${rebuild_pkg} == true ]]; then
    mv dist/* dist_old
    ./build_package.sh
fi
