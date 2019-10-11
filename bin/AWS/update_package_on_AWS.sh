#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $update_package_on_AWS [--key STR] [--address STR] [--framework STR] [--version FLOAT]

   optional arguments:
     -k, --key STR          path to ssh key
     -a, --address STR      ec2 instance Public DNS address
     -f, --framework STR    desired deep learning framework
     -v, --version FLOAT    AIToolbox version to be installed on ec2
     -h, --help             show this help message and exit

HEREDOC
}

key_path=
ec2_instance_address=
DL_framework="pytorch"
AIToolbox_version="0.3"

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

if [ "$key_path" == "" ] || [ "$ec2_instance_address" == "" ] || [ "$DL_framework" == "" ] || [ "$AIToolbox_version" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


if [ $DL_framework == "TF" ]; then
    py_env="tensorflow_p36"
elif [ $DL_framework == "pytorch" ]; then
    py_env="pytorch_p36"
else
    py_env="tensorflow_p36"
fi


scp -i $key_path ../../dist/aitoolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/project

ssh -i $key_path ec2-user@$ec2_instance_address "source activate $py_env ; pip uninstall aitoolbox ; pip install ~/project/aitoolbox-$AIToolbox_version.tar.gz"
