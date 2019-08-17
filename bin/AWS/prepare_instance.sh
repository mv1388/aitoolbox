#!/usr/bin/env bash

# Example how to run:
# ./prepare_instance.sh -k <SSH_KEY_LOCATION> -a <INSTANCE_IP_ADDRESS> -f pytorch -v 0.2 -p ~/PycharmProjects/Transformer -d SQuAD2 -r orig

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

   Usage: $prepare_instance [--key_path STR] [--address STR] [--project STR]

   optional arguments:
     -k, --key_path STR     path to ssh key
     -a, --address STR      ec2 instance Public DNS address
     -f, --framework STR    desired deep learning framework
     -v, --version FLOAT    AIToolbox version to be installed on ec2
     -p, --project STR      path to the project to be optionally uploaded to the running ec2 instance
     -d, --dataset STR      dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR      the preprocessed version of the main dataset
     -h, --help             show this help message and exit

HEREDOC
}

key_path=
ec2_instance_address=
DL_framework="pytorch"
AIToolbox_version="0.2"
local_project_path=
dataset_name=
preproc_dataset=

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -k|--key_path)
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


ssh -i "$key_path" ec2-user@"$ec2_instance_address" 'mkdir ~/project ; mkdir ~/project/data ; mkdir ~/project/model_results'

scp -i $key_path ../../dist/AIToolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/project
scp -i $key_path download_data.sh  ec2-user@$ec2_instance_address:~/project
scp -i $key_path run_experiment.sh  ec2-user@$ec2_instance_address:~/project

# Stuff for pyrouge package
scp -i $key_path ../pyrouge_set_rouge_path ec2-user@$ec2_instance_address:~/project
scp -i $key_path -r ../ROUGE-1.5.5 ec2-user@$ec2_instance_address:~/project

echo "#!/usr/bin/env bash

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo Ireland AWS zone: eu-west-1

aws configure
cd project

source activate $py_env

pip install -U boto3
pip install awscl
pip install -U numpy
pip install --ignore-installed greenlet

conda install -y -c conda-forge jsonnet
conda install -y -c anaconda seaborn=0.9.0

pip install AIToolbox-$AIToolbox_version.tar.gz

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


if [ $dataset_name != '' ]; then
    ./download_data.sh ~/project/data $dataset_name $preproc_dataset
fi
" > finish_prepare_instance.sh

chmod u+x finish_prepare_instance.sh
scp -i $key_path finish_prepare_instance.sh  ec2-user@$ec2_instance_address:~/.
rm finish_prepare_instance.sh

if [ "$local_project_path" != '' ]; then
    echo Uploading project folder $local_project_path
    source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address "~/project" $local_project_path
fi

ssh -i $key_path ec2-user@$ec2_instance_address
