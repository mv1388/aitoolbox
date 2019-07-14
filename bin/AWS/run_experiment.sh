#!/usr/bin/env bash

project_root_path=~/project
export PYTHONPATH=${PYTHONPATH}:$project_root_path

terminate_cmd=$1


source $project_root_path/AWS_run_scripts/AWS_core_scripts/aws_run_experiments_project.sh $project_root_path


if [ "$terminate_cmd" == "--terminate" ]; then
    echo Terminating the instance
    aws_instance_id=$(ec2-metadata --instance-id | cut -d " " -f 2)

    aws ec2 terminate-instances --instance-ids $aws_instance_id
fi
