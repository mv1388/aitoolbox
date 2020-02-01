#!/usr/bin/env bash

key_path=$1
ec2_instance_address=$2
shift 2


# usage function
function usage()
{
   cat << HEREDOC

   Usage: $ssh_to_instance <SSH_KEY_LOCATION> <INSTANCE_IP_ADDRESS>  (optional: [--ssh-tmux] [--os-name ubuntu])

   arguments:
     <SSH_KEY_LOCATION>     path to ssh key
     <INSTANCE_IP_ADDRESS>  ec2 instance Public DNS address
     -s, --ssh-tmux         if turned on attach to the running tmux session
     -o, --os-name STR      username depending on the OS chosen. Default is ubuntu
     -h, --help             show this help message and exit

HEREDOC
}

username="ubuntu"
ssh_to_tmux=false

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
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
