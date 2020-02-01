#!/usr/bin/env bash

# When you get ssh-ed to the instance finish the instance prep process by running:
# ./finish_prepare_instance.sh
# ./run_experiment.sh (optional: -t / --terminate)

project_root_path=~/project
export PYTHONPATH=${PYTHONPATH}:$project_root_path
export AWS_DEFAULT_REGION=eu-west-1

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $run_experiment (optional: [--terminate] [--experiment-script run_experiment.sh])

   optional arguments:
     -t, --terminate                the instance will be terminated when training is done
     -e, --experiment-script STR    name of the experiment bash script to be executed in order to start the training
     -h, --help                     show this help message and exit

HEREDOC
}

terminate_cmd=false
experiment_script_file="aws_run_experiments_project.sh"

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -t|--terminate)
    terminate_cmd=true
    shift 1 # past argument value
    ;;
    -e|--experiment-script)
    experiment_script_file="$2"
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


source $project_root_path/AWS_run_scripts/AWS_core_scripts/$experiment_script_file $project_root_path

if [[ $terminate_cmd == true ]]; then
    echo Terminating the instance
    aws_instance_id=$(ec2metadata --instance-id | cut -d " " -f 2)

    aws ec2 terminate-instances --instance-ids $aws_instance_id
fi
