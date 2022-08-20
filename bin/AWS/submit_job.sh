#!/usr/bin/env bash

# Example how to run:
# ./submit_job.sh -k <SSH_KEY_LOCATION> -p ~/PycharmProjects/QANet -d SQuAD2 -r orig

#######################

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./submit_job.sh [--project STR] ...

   arguments:
     -p, --project STR              path to the project to be optionally uploaded to the running ec2 instance

   optional arguments:
     -k, --key STR                  path to ssh key
     -n, --name STR                 name for the created instance
     -d, --dataset STR              dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR              the preprocessed version of the main dataset
     -f, --framework STR            desired deep learning framework
     -v, --version FLOAT            AIToolbox version to be installed on ec2
     -i, --instance-config STR      instance configuration json filename
     --instance-type STR            instance type label; if this is provided the value from --instance-config is ignored
     -e, --experiment-script STR    name of the experiment bash script to be executed in order to start the training
     --default-log                  if used than the logs will be saved to the default log file training.log without any timestamps
     --log-s3-upload-dir STR        path to the logs folder on S3 to which the training log should be uploaded
     -o, --os-name STR              username depending on the OS chosen. Default is ubuntu
     -t, --terminate                the instance will be terminated when training is done
     -s, --ssh-start                automatically ssh into the instance when the training starts
     --without-scheduler            run experiment without the training job scheduler
     --on-demand                    create on-demand instance instead of spot instance
     --central-region               create the instance in the central region (Frankfurt)
     --pypi                         install package from PyPI instead of the local package version
     -h, --help                     show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
local_project_path="None"
dataset_name="None"
preproc_dataset="None"
DL_framework="pytorch"
AIToolbox_version="1.7.0"
instance_config="default_config.json"
instance_type=
experiment_script_file="aws_run_experiments_project.sh"
log_s3_dir_path="s3://model-result/training_logs"
default_log=false
username="ubuntu"
terminate_cmd=false
ssh_at_start=false
run_with_scheduler=true
spot_instance=true
aws_region="eu-west-1"
local_pypi_install=""
instance_name=

default_logging_filename="training.log"

job_timestamp=$(date +"%Y%m%d_%H_%M_%S")
logging_filename="training_$job_timestamp.log"
logging_path="~/project/$logging_filename"


while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
    shift 2 # past argument value
    ;;
    -n|--instance_name)
    instance_name="$2"
    shift 2 # past argument value
    ;;
    -p|--project)
    local_project_path="$2"
    shift 2 # past argument value
    ;;
    -d|--dataset)
    dataset_name="$2"
    shift 2 # past argument value
    ;;
    -r|--preproc)
    preproc_dataset="$2"
    shift 2 # past argument value
    ;;
    -f|--framework)
    DL_framework="$2"
    shift 2 # past argument value
    ;;
    -v|--version)
    AIToolbox_version="$2"
    shift 2 # past argument value
    ;;
    -i|--instance-config)
    instance_config="$2"
    shift 2 # past argument value
    ;;
    --instance-type)
    instance_type="$2"
    shift 2 # past argument value
    ;;
    -e|--experiment-script)
    experiment_script_file="$2"
    shift 2 # past argument value
    ;;
    --default-log)
    default_log=true
    shift 1 # past argument value
    ;;
    --log-s3-upload-dir)
    log_s3_dir_path="$2"
    shift 2 # past argument value
    ;;
    -o|--os-name)
    username="$2"
    shift 2 # past argument value
    ;;
    -t|--terminate)
    terminate_cmd=true
    shift 1 # past argument value
    ;;
    -s|--ssh-start)
    ssh_at_start=true
    shift 1 # past argument value
    ;;
    --without-scheduler)
    run_with_scheduler=false
    shift 1 # past argument value
    ;;
    --on-demand)
    spot_instance=false
    shift 1 # past argument value
    ;;
    --central-region)
    aws_region="eu-central-1"
    shift 1 # past argument value
    ;;
    --pypi)
    local_pypi_install="--pypi"
    shift 1 # past argument value
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

if [ "$local_project_path" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi

if [ "$DL_framework" == "TF" ]; then
    py_env="tensorflow"
elif [ "$DL_framework" == "pytorch" ]; then
    py_env="pytorch"
else
    py_env="pytorch"
fi

terminate_setting=""
if [ "$terminate_cmd" == true ]; then
    terminate_setting="--terminate"
fi

if [[ "$instance_type" != "" ]]; then
    instance_type="--instance-type $instance_type"
fi

if [ "$aws_region" == "eu-central-1" ]; then
    instance_config=${instance_config%.*}_central.json
fi

if [ "$default_log" == true ]; then
    logging_filename=$default_logging_filename
    logging_path="~/project/$logging_filename"
fi

log_upload_setting=""
if [ "$log_s3_dir_path" != "None" ] && [ "$log_s3_dir_path" != "False" ]; then
    log_upload_setting="--log-path $logging_path --log-s3-upload-dir $log_s3_dir_path"
fi


# Set the region either to Ireland or Frankfurt
export AWS_DEFAULT_REGION=$aws_region

#############################
# Instance creation
#############################
spot_instance_option=""
if [ "$spot_instance" == true ]; then
    echo "Creating spot instance"
    spot_instance_option=(--instance-market-options '{ "MarketType": "spot" }')
else
    echo "Creating on-demand instance"
fi

instance_id=$(aws ec2 run-instances $instance_type "${spot_instance_option[@]}" --cli-input-json file://configs/$instance_config --query 'Instances[0].InstanceId' --output text)

if [[ "$instance_name" != "" ]]; then
    aws ec2 create-tags --resources $instance_id --tags Key=Name,Value=$instance_name
fi

echo "Waiting for instance create"
aws ec2 wait instance-status-ok --instance-ids $instance_id

ec2_instance_address=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[*].Instances[*].PublicDnsName' --output text)
ec2_instance_ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[*].Instances[*].PublicIpAddress' --output text)


##############################
# Preparing the instance
##############################
echo "Preparing instance"
./prepare_instance.sh -k $key_path -a $ec2_instance_address \
    -f $DL_framework -v $AIToolbox_version -p $local_project_path -d $dataset_name -r $preproc_dataset -o $username --aws-region $aws_region $local_pypi_install --no-ssh


#########################################################
# Bootstrapping the instance and execute the training
#########################################################
printf "\n========================================================\n"
echo "Running the job"
if [ "$run_with_scheduler" == true ]; then
  ssh -i $key_path $username@$ec2_instance_address \
      "source activate $py_env ;\
      tmux new-session -d -s 'training' \
      'source finish_prepare_instance.sh ;\
        cd project ;\
        python ~/training_job_scheduler.py add-job --experiment-script $experiment_script_file ;\
        python ~/training_job_scheduler.py run $terminate_setting $log_upload_setting --aws-region $aws_region' \
      \; \
      pipe-pane 'cat > $logging_path'"

else
  ssh -i $key_path $username@$ec2_instance_address \
      "source activate $py_env ;\
      tmux new-session -d -s 'training' \
      'source finish_prepare_instance.sh ;\
        cd project ;\
        ./run_experiment.sh $terminate_setting --experiment-script $experiment_script_file $log_upload_setting --cleanup-script --aws-region $aws_region' \
      \; \
      pipe-pane 'cat > $logging_path'"
fi

echo "Instance DNS address: $ec2_instance_address"
echo "Instance IP address: $ec2_instance_ip_address"
echo "Instance AWS ID: $instance_id"
echo "To easily ssh connect into the running job session execute:"
echo
echo "    ./ssh_to_instance.sh $ec2_instance_address -s"
echo
echo
echo "To terminate the instance execute:"
echo
echo "    aws ec2 terminate-instances --instance-ids $instance_id"
echo

if [ $ssh_at_start == true ]; then
    ./ssh_to_instance.sh $ec2_instance_address --os-name $username --ssh-tmux
fi
