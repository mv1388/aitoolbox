#!/usr/bin/env bash

ec2_instance_address=$1
shift 1

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./ssh_to_instance.sh <INSTANCE_IP_ADDRESS>  (optional: [--key <SSH_KEY_LOCATION>] [--ssh-tmux] [--os-name ubuntu])

   arguments:
     <INSTANCE_IP_ADDRESS>  ec2 instance Public DNS address

   optional arguments:
     -k, --key STR          path to ssh key
     -s, --ssh-tmux         if turned on attach to the running tmux session
     -o, --os-name STR      username depending on the OS chosen. Default is ubuntu
     -h, --help             show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
username="ubuntu"
ssh_to_tmux=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key)
    key_path="$2"
    shift 2 # past argument value
    ;;
    -o|--os-name)
    username="$2"
    shift 2 # past argument value
    ;;
    -s|--ssh-tmux)
    ssh_to_tmux=true
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


if [ $ssh_to_tmux == false ]; then
    ssh -i $key_path $username@$ec2_instance_address
else
    ssh -i $key_path $username@$ec2_instance_address -t "tmux a -t training"
fi
