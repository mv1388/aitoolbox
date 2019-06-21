#!/usr/bin/env bash

key_path=$1
ec2_instance_address=$2

local_project_path=$3

echo Re-ploading project folder $local_project_path

source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address "~/project" $local_project_path
