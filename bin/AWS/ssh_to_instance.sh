#!/usr/bin/env bash

key_path=$1
ec2_instance_address=$2

ssh -i $key_path ubuntu@$ec2_instance_address
