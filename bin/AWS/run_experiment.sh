#!/usr/bin/env bash

# When you get ssh-ed to the instance finish the instance prep process by running:
# ./finish_prepare_instance.sh
# ./run_experiment.sh (optional: -t / --terminate)


# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./run_experiment.sh (optional: [--terminate] [--experiment-script run_experiment.sh])

   optional arguments:
     -t, --terminate                the instance will be terminated when training is done
     -e, --experiment-script STR    name of the experiment bash script to be executed in order to start the training
     -p, --project-root STR         path to the project root on the execution server/AWS
     -l, --log-path STR             path to the local log file which will be uploaded to s3
     --log-s3-upload-dir STR        path to the logs folder on S3 to which the training log should be uploaded
     -i, --log-iteration INT        index of an executed job via the scheduler
     -c, --cleanup-script           post execution cleanup script
     --aws-region STR               create the instance in the specified region. Default is Ireland (eu-west-1)
     -h, --help                     show this help message and exit

HEREDOC
}

terminate_cmd=false
experiment_script_file="aws_run_experiments_project.sh"
project_root_path=~/project
log_file_path=
log_s3_dir_path="s3://model-result/training_logs"
log_iteration=-1
post_experiment_run_cleanup=false
aws_region="eu-west-1"

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
    -p|--project-root)
    project_root_path="$2"
    shift 2 # past argument value
    ;;
    -l|--log-path)
    log_file_path="$2"
    shift 2 # past argument value
    ;;
    --log-s3-upload-dir)
    log_s3_dir_path="$2"
    shift 2 # past argument value
    ;;
    -i|--log-iteration)
    log_iteration="$2"
    shift 2 # past argument value
    ;;
    -c|--cleanup-script)
    post_experiment_run_cleanup=true
    shift 1 # past argument value
    ;;
    --aws-region)
    aws_region="$2"
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


export PYTHONPATH=${PYTHONPATH}:$project_root_path

# Set the region either to Ireland or Frankfurt
export AWS_DEFAULT_REGION=$aws_region


echo "
************************************************************
***                                                      ***
***              STARTING THE TRAINING JOB               ***
***                                                      ***
************************************************************
"


source $project_root_path/AWS_run_scripts/AWS_core_scripts/$experiment_script_file $project_root_path


if [[ $log_file_path != "" && -f $log_file_path ]]; then
    filtered_log_file_path="$(dirname $log_file_path)/filtered_$(basename $log_file_path)"

    if [ "$log_iteration" -ge 0 ]; then
        log_file_path_without_ext="${log_file_path%.*}"
        extension="${log_file_path##*.}"
        log_file_path_iteration="${log_file_path_without_ext}_train_job_${log_iteration}.${extension}"

        cp $log_file_path $log_file_path_iteration
        log_file_path=$log_file_path_iteration

        filtered_log_file_path_without_ext="${filtered_log_file_path%.*}"
        filtered_log_file_path="${filtered_log_file_path_without_ext}_train_job_${log_iteration}.${extension}"
    fi

    sed -n -e '/              STARTING THE TRAINING JOB               /, $p' $log_file_path | grep -v '%|.*|' > $filtered_log_file_path

    s3_log_path="$log_s3_dir_path/$(basename $log_file_path)"
    aws s3 cp $log_file_path $s3_log_path

    s3_filtered_log_path="$log_s3_dir_path/$(basename $filtered_log_file_path)"
    aws s3 cp $filtered_log_file_path $s3_filtered_log_path
fi

if [[ $post_experiment_run_cleanup == true ]]; then
    echo Running post execution cleanup script
    source $project_root_path/AWS_run_scripts/AWS_bootstrap/post_run_cleanup.sh $project_root_path
fi

if [[ $terminate_cmd == true ]]; then
    echo Terminating the instance
    aws_instance_id=$(ec2metadata --instance-id | cut -d " " -f 2)

    aws ec2 terminate-instances --instance-ids $aws_instance_id
fi
