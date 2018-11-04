#!/usr/bin/env bash


key_path=$1
ec2_instance_address=$2

DL_framework=$3

AIToolbox_version=$4


if [ $DL_framework == "TF" ]; then
    py_env="tensorflow_p36"
elif [ $DL_framework == "pytorch" ]; then
    py_env="pytorch_p36"
else
    py_env="tensorflow_p36"
fi


scp -i $key_path ../../dist/AIToolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/project

ssh -i $key_path ec2-user@$ec2_instance_address "source activate $py_env ; pip uninstall AIToolbox ; pip install ~/project/AIToolbox-$AIToolbox_version.tar.gz"
