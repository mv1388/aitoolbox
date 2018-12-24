#!/usr/bin/env bash

project_root_path=~/project
export PYTHONPATH=${PYTHONPATH}:$project_root_path

terminate_cmd=$1
aws_instance_id=$2

# TODO: add tmux

source $project_root_path/AWS_run_scripts/AWS_core_scripts/aws_run_experiments_project.sh $project_root_path


if [ "$terminate_cmd" == "--terminate" ]; then
    echo Terminating the instance
    aws ec2 terminate-instances --instance-ids $aws_instance_id
fi
