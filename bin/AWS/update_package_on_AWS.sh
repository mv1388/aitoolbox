#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./update_package_on_AWS.sh [--address STR] [--framework STR] [--version FLOAT]

   arguments:
     -a, --address STR      ec2 instance Public DNS address

   optional arguments:
     -f, --framework STR    desired deep learning framework
     -v, --version FLOAT    AIToolbox version to be installed on ec2
     -k, --key STR          path to ssh key
     -o, --os-name STR      username depending on the OS chosen. Default is ubuntu
     -h, --help             show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
ec2_instance_address=
DL_framework="pytorch"
AIToolbox_version="1.8.0"
username="ubuntu"

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
    -f|--framework)
    DL_framework="$2"
    shift 2 # past argument value
    ;;
    -v|--version)
    AIToolbox_version="$2"
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

if [ "$ec2_instance_address" == "" ] || [ "$DL_framework" == "" ] || [ "$AIToolbox_version" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


if [ $DL_framework == "TF" ]; then
    py_env="tensorflow"
elif [ $DL_framework == "pytorch" ]; then
    py_env="pytorch"
else
    py_env="pytorch"
fi


scp -i $key_path ../../dist/aitoolbox-$AIToolbox_version.tar.gz  $username@$ec2_instance_address:~/project

ssh -i $key_path $username@$ec2_instance_address \
    "source activate $py_env ;\
      pip uninstall aitoolbox ;\
      pip install ~/project/aitoolbox-$AIToolbox_version.tar.gz"
