#!/usr/bin/env bash

# Example how to run two most common use-cases:
#   Single GPU tests:
#       ./test_core_pytorch_compare.sh
#       Or to speed it up:
#       ./test_core_pytorch_compare.sh --instance-type p3.2xlarge
#
#   Multi GPU tests:
#       ./test_core_pytorch_compare.sh --multi-gpu --instance-type p2.8xlarge
#       Or to speed it up:
#       ./test_core_pytorch_compare.sh --multi-gpu --instance-type p3.8xlarge


# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./test_core_pytorch_compare.sh ...

   Example how to run two most common use-cases:
       Single GPU tests:
           ./test_core_pytorch_compare.sh
           Or to speed it up:
           ./test_core_pytorch_compare.sh --instance-type p3.2xlarge

       Multi GPU tests:
           ./test_core_pytorch_compare.sh --multi-gpu --instance-type p2.8xlarge
           Or to speed it up:
           ./test_core_pytorch_compare.sh --multi-gpu --instance-type p3.8xlarge

   optional arguments:
     --multi, --multi-gpu           execute tests in the multi GPU setting instead of the default single GPU
     --instance-type STR            instance type label; if this is provided the value from --instance-config is ignored
     -i, --instance-config STR      instance configuration json filename
     --no-ssh                       after test job is submitted don't automatically ssh into the running instance
     -k, --key STR                  path to ssh key
     -o, --os-name STR              username depending on the OS chosen. Default is ubuntu
     --on-demand                    create on-demand instance instead of spot instance
     --central-region               create the instance in the central region (Frankfurt)
     -h, --help                     show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
instance_config="default_config.json"
instance_type=
username="ubuntu"
py_env="pytorch_latest_p36"
ssh_at_start=true
spot_instance=true
aws_region="eu-west-1"

gpu_mode="single"

job_timestamp=$(date +"%Y%m%d_%H_%M_%S")
logging_filename="comparison_test_$job_timestamp.log"
logging_path="~/package_test/$logging_filename"


while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
    shift 2 # past argument value
    ;;
    --multi|--multi-gpu)
    gpu_mode="multi"
    shift 1 # past argument value
    ;;
    -i|--instance-config)
    instance_config="$2"
    shift 2 # past argument value
    ;;
    --instance-type)
    instance_type="$2"
    shift 2 # past argument value
    ;;
    --no-ssh)
    ssh_at_start=false
    shift 1 # past argument value
    ;;
    -o|--os-name)
    username="$2"
    shift 2 # past argument value
    ;;
    --on-demand)
    spot_instance=false
    shift 1 # past argument value
    ;;
    --central-region)
    aws_region="eu-central-1"
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

if [[ "$instance_type" != "" ]]; then
    instance_type="--instance-type $instance_type"
fi

if [ "$aws_region" == "eu-central-1" ]; then
    instance_config=${instance_config%.*}_central.json
fi

pytest_dir="tests_gpu/test_single_gpu"
if [[ "$gpu_mode" == "multi" ]]; then
    pytest_dir="tests_gpu/test_multi_gpu"
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

echo "Waiting for instance create"
aws ec2 wait instance-status-ok --instance-ids $instance_id

ec2_instance_address=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[*].Instances[*].PublicDnsName' --output text)


##############################
# Preparing the instance
##############################
echo "Preparing instance"
ssh -i $key_path -o "StrictHostKeyChecking no" $username@$ec2_instance_address 'mkdir ~/package_test'

scp -i $key_path -r ../../aitoolbox $username@$ec2_instance_address:~/package_test
scp -i $key_path -r ../../tests_gpu $username@$ec2_instance_address:~/package_test
scp -i $key_path ../../requirements.txt $username@$ec2_instance_address:~/package_test


#########################################################
# Bootstrapping the instance and execute the testing
#########################################################
echo "Running the comparison tests"
ssh -i $key_path $username@$ec2_instance_address \
    "source activate $py_env ; tmux new-session -d -s 'training' 'export AWS_DEFAULT_REGION=$aws_region ; cd package_test ; pip install pytest seaborn==0.9.0 ; pip install -r requirements.txt ; pip install torchtext==0.7 torchvision>=0.9.1 ; pip install --upgrade protobuf ; pytest $pytest_dir -s ; aws s3 cp $logging_path s3://aitoolbox-testing/core_pytorch_comparisson_testing/$logging_filename ; aws ec2 terminate-instances --instance-ids $instance_id' \; pipe-pane 'cat > $logging_path'"

echo "Instance IP: $ec2_instance_address"

if [[ ${ssh_at_start} == true ]]; then
    ./ssh_to_instance.sh $ec2_instance_address --os-name $username --ssh-tmux
fi
