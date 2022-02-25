#!/usr/bin/env bash

#######################

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./create_instance.sh [--project STR] ...

   optional arguments:
     -k, --key STR                  path to ssh key
     -p, --project STR              path to the project to be optionally uploaded to the running ec2 instance
     -d, --dataset STR              dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR              the preprocessed version of the main dataset
     -f, --framework STR            desired deep learning framework
     -v, --version FLOAT            AIToolbox version to be installed on ec2
     -i, --instance-config STR      instance configuration json filename
     --instance-type STR            instance type label; if this is provided the value from --instance-config is ignored
     -n, --no-bootstrap             keep the newly created instance and don't run the bootstrapping
     -o, --os-name STR              username depending on the OS chosen. Default is ubuntu
     -t, --terminate                the instance will be terminated when training is done
     -s, --ssh-start                automatically ssh into the instance when the training starts
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
AIToolbox_version="1.4.0"
instance_config="config_p2_xlarge.json"
instance_type=
run_bootstrap=true
username="ubuntu"
terminate_cmd=false
ssh_at_start=false
aws_region="eu-west-1"
local_pypi_install=""


while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
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
    -n|--no-bootstrap)
    run_bootstrap=false
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

if [ "$key_path" == "" ] ; then
    echo "Not provided required parameters"
    usage
    exit
fi

if [ "$DL_framework" == "TF" ]; then
    py_env="tensorflow_p36"
elif [ "$DL_framework" == "pytorch" ]; then
    py_env="pytorch_latest_p36"
else
    py_env="pytorch_latest_p36"
fi

terminate_setting=""
if [ "$terminate_cmd" == true ]; then
    terminate_setting="--terminate"
fi

if [[ "$instance_type" != "" ]]; then
    instance_config=config_$(tr . _ <<< $instance_type).json
fi

if [ "$aws_region" == "eu-central-1" ]; then
    instance_config=${instance_config%.*}_central.json
fi


#############################
# Instance creation
#############################
echo "Creating spot request"
request_id=$(aws ec2 request-spot-instances --launch-specification file://configs/$instance_config --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)
aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $request_id

echo "Waiting for instance create"
instance_id=$(aws ec2 describe-spot-instance-requests --spot-instance-request-ids $request_id --query 'SpotInstanceRequests[0].InstanceId' --output text)
aws ec2 wait instance-status-ok --instance-ids $instance_id

ec2_instance_address=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[*].Instances[*].PublicDnsName' --output text)


##############################
# Preparing the instance
##############################
echo "Preparing instance"
./prepare_instance.sh -k $key_path -a $ec2_instance_address \
    -f $DL_framework -v $AIToolbox_version -p $local_project_path -d $dataset_name -r $preproc_dataset -o $username --aws-region $aws_region $local_pypi_install --no-ssh


#########################################################
# Bootstrapping the instance and execute the training
#########################################################
if [ $run_bootstrap == true ]; then
    echo "Instance bootstrapping"
    ssh -i $key_path $username@$ec2_instance_address \
        "source activate $py_env ; tmux new-session -d -s 'training' './finish_prepare_instance.sh'"
fi

echo "Instance IP: $ec2_instance_address"

if [ $ssh_at_start == true ]; then
    ./ssh_to_instance.sh $ec2_instance_address --os-name $username --ssh-tmux
fi
