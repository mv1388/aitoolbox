#!/usr/bin/env bash

key_path=$1
ec2_instance_address=$2

if [[ $3 == "" ]]; then
    username="ubuntu"
else
    username=$3
fi

ssh -i $key_path $username@$ec2_instance_address
