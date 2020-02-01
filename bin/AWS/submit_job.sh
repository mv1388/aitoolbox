#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./submit_job.sh [--key STR] [--project STR] ...

   arguments:
     -k, --key STR          path to ssh key
     -p, --project STR      path to the project to be optionally uploaded to the running ec2 instance

   optional arguments:
     -f, --framework STR    desired deep learning framework
     -v, --version FLOAT    AIToolbox version to be installed on ec2
     -d, --dataset STR      dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR      the preprocessed version of the main dataset
     -x, --apex             switch on to install Nvidia Apex library for mixed precision training
     -o, --os-name STR      username depending on the OS chosen. Default is ubuntu
     -t, --terminate        the instance will be terminated when training is done
     -s, --ssh-start        automatically ssh into the instance when the training starts
     -h, --help             show this help message and exit

HEREDOC
}

key_path=
DL_framework="pytorch"
AIToolbox_version="0.3"
local_project_path=
dataset_name=
preproc_dataset=
use_apex=false
username="ubuntu"
terminate_cmd=false
ssh_at_start=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
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
    -x|--apex)
    use_apex=true
    shift 1 # past argument value
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

if [ "$key_path" == "" ] || [ "$local_project_path" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi

if [ "$DL_framework" == "TF" ]; then
    py_env="tensorflow_p36"
elif [ "$DL_framework" == "pytorch" ]; then
    py_env="pytorch_p36"
else
    py_env="tensorflow_p36"
fi

apex_setting=""
if [ "$use_apex" == true ]; then
    apex_setting="--apex"
fi

terminate_setting=""
if [ "$terminate_cmd" == true ]; then
    terminate_setting="--terminate"
fi


#############################
# Instance creation
#############################
echo "Creating spot request"
request_id=$(aws ec2 request-spot-instances --launch-specification file://configs/config_spec.json --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)
aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $request_id

echo "Waiting for instance create"
instance_id=$(aws ec2 describe-spot-instance-requests --spot-instance-request-ids $request_id --query 'SpotInstanceRequests[0].InstanceId' --output text)
aws ec2 wait instance-status-ok --instance-ids $instance_id

ec2_instance_address=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[*].Instances[*].PublicDnsName' --output text)


########################################################
# Bootstrapping the instance and executing the training
########################################################
echo "Preparing instance"
./prepare_instance.sh -k $key_path -a $ec2_instance_address \
    -f $DL_framework -v $AIToolbox_version -p $local_project_path -d $dataset_name -r $preproc_dataset $apex_setting -o $username --no-ssh

echo "Running the job"
ssh -i $key_path $username@$ec2_instance_address \
    "./finish_prepare_instance.sh ; source activate $py_env ; cd project ; tmux new-session -d -s 'training' ./run_experiment.sh $terminate_setting"

echo "Instance IP: $ec2_instance_address"

if [ $ssh_at_start == true ]; then
    ./ssh_to_instance.sh $key_path $ec2_instance_address --os-name $username --ssh-tmux
fi
