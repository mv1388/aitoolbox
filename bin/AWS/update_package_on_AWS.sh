#!/usr/bin/env bash


key_path=$1
ec2_instance_address=$2

AIToolbox_version=$3

scp -i $key_path ../../dist/AIToolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/project

ssh -i $key_path ec2-user@$ec2_instance_address "source activate tensorflow_p36 ; pip uninstall AIToolbox ; pip install ~/project/AIToolbox-$AIToolbox_version.tar.gz"
