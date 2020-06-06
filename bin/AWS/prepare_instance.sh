#!/usr/bin/env bash

# Example how to run:
# ./prepare_instance.sh -k <SSH_KEY_LOCATION> -a <INSTANCE_IP_ADDRESS> -f pytorch -v 1.1.0 -p ~/PycharmProjects/Transformer -d SQuAD2 -r orig

# When you get ssh-ed to the instance finish the instance prep process by running:
# ./finish_prepare_instance.sh
# ./run_experiment.sh (optional: --terminate)

#######################

# Args: specify one of the available datasets
#   Download the data from S3 or from local computer

# Create experiment folder structure
# Upload the AIToolbox package and install it
# Upload specified project code

#######################

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./prepare_instance.sh [--address STR] [--project STR] ...

   arguments:
     -a, --address STR      ec2 instance Public DNS address
     -f, --framework STR    desired deep learning framework
     -v, --version FLOAT    AIToolbox version to be installed on ec2

   optional arguments:
     -k, --key STR          path to ssh key
     -p, --project STR      path to the project to be optionally uploaded to the running ec2 instance
     -d, --dataset STR      dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR      the preprocessed version of the main dataset
     -x, --apex             switch on to install Nvidia Apex library for mixed precision training
     --deepspeed            install Microsoft DeepSpeed library
     -o, --os-name STR      username depending on the OS chosen. Default is ubuntu
     --no-ssh               disable auto ssh-ing to the instance
     -h, --help             show this help message and exit

HEREDOC
}

key_path=$(jq -r '.key_path' configs/my_config.json)
ec2_instance_address=
DL_framework="pytorch"
AIToolbox_version="1.1.0"
local_project_path="None"
dataset_name="None"
preproc_dataset="None"
use_apex=false
use_deepspeed=false
username="ubuntu"
auto_ssh_to_instance=true

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
    -p|--project)
    local_project_path="$2"
    shift 2 # past argument value
    ;;
    -d|--dataset)
    dataset_name="$2"
    shift 2 # past argument value
    ;;
    -r|--preproc)
    preproc_dataset="$2"
    shift 2 # past argument value
    ;;
    -x|--apex)
    use_apex=true
    shift 1 # past argument value
    ;;
    --deepspeed)
    use_deepspeed=true
    shift 1 # past argument value
    ;;
    -o|--os-name)
    username="$2"
    shift 2 # past argument value
    ;;
    --no-ssh)
    auto_ssh_to_instance=false
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

if [ "$key_path" == "" ] || [ "$ec2_instance_address" == "" ] || [ "$DL_framework" == "" ] || [ "$AIToolbox_version" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


if [ "$DL_framework" == "TF" ]; then
    py_env="tensorflow_p36"
elif [ "$DL_framework" == "pytorch" ]; then
    py_env="pytorch_p36"
else
    py_env="tensorflow_p36"
fi


ssh -i $key_path -o "StrictHostKeyChecking no" $username@$ec2_instance_address 'mkdir ~/project ; mkdir ~/project/data ; mkdir ~/project/model_results'

scp -i $key_path ../../dist/aitoolbox-$AIToolbox_version.tar.gz  $username@$ec2_instance_address:~/project
scp -i $key_path download_data.sh  $username@$ec2_instance_address:~/project
scp -i $key_path run_experiment.sh  $username@$ec2_instance_address:~/project

# Stuff for pyrouge package
scp -i $key_path ../pyrouge_set_rouge_path $username@$ec2_instance_address:~/project
scp -i $key_path -r ../ROUGE-1.5.5 $username@$ec2_instance_address:~/project

echo "#!/usr/bin/env bash

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export AWS_DEFAULT_REGION=eu-west-1

#echo Ireland AWS zone: eu-west-1

#aws configure
cd project

source activate $py_env

pip install -U boto3
pip install awscli
pip install -U numpy
pip install --ignore-installed greenlet
pip install jsonnet seaborn==0.9.0

#conda install -y -c conda-forge jsonnet
#conda install -y -c anaconda seaborn=0.9.0

if [ $use_apex == true ]; then
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./
    cd ..
fi

if [ $use_deepspeed == true ]; then
    git clone https://github.com/microsoft/DeepSpeed
    cd DeepSpeed
    ./install.sh --local_only --skip_requirements
    cd ..
fi

pip install aitoolbox-$AIToolbox_version.tar.gz

if [ $local_project_path != 'None' ]; then
    pip install -r ~/project/AWS_run_scripts/AWS_bootstrap/requirements.txt
    ~/project/AWS_run_scripts/AWS_bootstrap/bootstrap.sh
fi

#./pyrouge_set_rouge_path ~/project/ROUGE-1.5.5
#
#sudo yum -y install perl-CPAN
##sudo perl -MCPAN -e 'install LWP::UserAgent::Cached'
##sudo perl -MCPAN -e 'install Bundle::LWP'
#sudo yum install -y perl-libwww-perl
#sudo perl -MCPAN -e 'install DB_File'
#
#cd ROUGE-1.5.5/data
#rm WordNet-2.0.exc.db
#cd WordNet-2.0-Exceptions
#./buildExeptionDB.pl . exc WordNet-2.0.exc.db
#cd ../
#ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
#cd ../..


if [ $dataset_name != 'None' ]; then
    ./download_data.sh -p ~/project/data -d $dataset_name -r $preproc_dataset
fi
" > finish_prepare_instance.sh

chmod u+x finish_prepare_instance.sh
scp -i $key_path finish_prepare_instance.sh  $username@$ec2_instance_address:~/.
rm finish_prepare_instance.sh

if [ "$local_project_path" != 'None' ]; then
    echo Uploading project folder $local_project_path
    source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address "~/project" $local_project_path
fi

if [ $auto_ssh_to_instance == true ]; then
    ssh -i $key_path $username@$ec2_instance_address
fi
