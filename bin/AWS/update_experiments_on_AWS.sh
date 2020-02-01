#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $update_experiments_on_AWS [--address STR] [--project STR]

   arguments:
     -a, --address STR      ec2 instance Public DNS address
     -p, --project STR      path to the project to be optionally uploaded to the running ec2 instance
     -h, --help             show this help message and exit

   optional arguments:
     -k, --key STR          path to ssh key

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
ec2_instance_address=
local_project_path=

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

source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address "~/project" $local_project_path
