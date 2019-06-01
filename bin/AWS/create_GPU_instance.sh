#!/usr/bin/env bash

# https://aws.amazon.com/blogs/machine-learning/train-deep-learning-models-on-gpus-using-amazon-ec2-spot-instances/


 aws ec2 run-instances \
    --image-id ami-0061ecbbc1cbd45f5 \
    --security-group-ids <SECURITY_GROUP_ID> \
    --count 1 \
    --instance-type p2.xlarge \
    --key-name <KEYPAIR_NAME> \
    --subnet-id <SUBNET_ID> \
    --query "Instances[0].InstanceId"


aws ec2 create-volume \
    --size 30 \
    --region eu-west-1 \
    --availability-zone eu-west-1b \
    --volume-type gp2 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DL-datasets-checkpoints}]'


aws ec2 attach-volume \
    --volume-id <VOLUME_ID> \
    --instance-id <INSTANCE_ID> \
    --device /dev/sdf
