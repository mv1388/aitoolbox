#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./update_experiments_on_AWS.sh [--address STR] [--project STR]

   arguments:
     -a, --address STR        ec2 instance Public DNS address
     -p, --project STR        path to the project to be optionally uploaded to the running ec2 instance
     -h, --help               show this help message and exit

   optional arguments:
     -k, --key STR            path to ssh key
     --add-job                add training job to the running training job scheduler
     --experiment-script STR  name of the experiment bash script to be executed in order to start the training
     --aws-project-root STR   path to the aws-based project root

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
ec2_instance_address=
local_project_path=

add_training_scheduler_job=false
experiment_script_file="aws_run_experiments_project.sh"
aws_project_root_path=~/project

username="ubuntu"
py_env="pytorch"


while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
    shift 2 # past argument value
    ;;
    -a|--address)
    ec2_instance_address="$2"
    shift 2 # past argument value
    ;;
    -p|--project)
    local_project_path="$2"
    shift 2 # past argument value
    ;;
    --add-job)
    add_training_scheduler_job=true
    shift 1 # past argument value
    ;;
    -e|--experiment-script)
    experiment_script_file="$2"
    shift 2 # past argument value
    ;;
    --aws-project-root)
    aws_project_root_path="$2"
    shift 2 # past argument value
    ;;
    -h|--help )
    usage;
    exit;
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    usage;
    exit;
    ;;
esac
done

if [ "$key_path" == "" ] || [ "$ec2_instance_address" == "" ] || [ "$local_project_path" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


echo Re-ploading project folder $local_project_path

source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address $aws_project_root_path $local_project_path

if [ $add_training_scheduler_job == true ]; then
    ssh -i $key_path $username@$ec2_instance_address "source activate $py_env ; python ~/training_job_scheduler.py add-job --experiment-script $experiment_script_file --project-root $aws_project_root_path"
fi
