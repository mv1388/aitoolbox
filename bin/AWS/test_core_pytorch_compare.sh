#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./test_core_pytorch_compare.sh ...

   optional arguments:
     -k, --key STR                  path to ssh key
     -i, --instance-config STR      instance configuration json filename
     --instance-type STR            instance type label; if this is provided the value from --instance-config is ignored
     -o, --os-name STR              username depending on the OS chosen. Default is ubuntu
     -h, --help                     show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
instance_config="config_p2_xlarge.json"
instance_type=
username="ubuntu"
py_env="pytorch_p36"

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
    -i|--instance-config)
    instance_config="$2"
    shift 2 # past argument value
    ;;
    --instance-type)
    instance_type="$2"
    shift 2 # past argument value
    ;;
    -o|--os-name)
    username="$2"
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

if [[ "$instance_type" != "" ]]; then
    instance_config=config_$(tr . _ <<< $instance_type).json
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
ssh -i $key_path -o "StrictHostKeyChecking no" $username@$ec2_instance_address 'mkdir ~/package_test'

scp -i $key_path -r ../../aitoolbox $username@$ec2_instance_address:~/package_test
scp -i $key_path -r ../../tests_core_pytorch_compare $username@$ec2_instance_address:~/package_test
scp -i $key_path ../../requirements.txt $username@$ec2_instance_address:~/package_test


#########################################################
# Bootstrapping the instance and execute the testing
#########################################################
echo "Running the comparison tests"
ssh -i $key_path $username@$ec2_instance_address \
    "source activate $py_env ; tmux new-session -d -s 'training' 'cd package_test ; pip install moto pytest ; pip install -r requirements.txt ; pytest tests_core_pytorch_compare -s ; aws s3 cp $logging_path s3://aitoolbox-testing/core_pytorch_comparisson_testing/$logging_filename ; aws ec2 terminate-instances --instance-ids $instance_id' \; pipe-pane 'cat > $logging_path'"

echo "Instance IP: $ec2_instance_address"

./ssh_to_instance.sh $ec2_instance_address --os-name $username --ssh-tmux
