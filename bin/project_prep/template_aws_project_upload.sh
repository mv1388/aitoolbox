#!/usr/bin/env bash

key_path=$1
ec2_instance_address=$2

project_root_path=$3
local_project_path=$4


# AWS_run_scripts
# Important: AWS_core_scripts sub-folder needs to be uploaded
scp -i $key_path -r $local_project_path/AWS_run_scripts ubuntu@$ec2_instance_address:$project_root_path/


##### ADD OTHER FOLDERS TO BE UPLOADED #######

# scp -i $key_path -r $local_project_path/ . . . .

