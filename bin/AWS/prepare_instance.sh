#!/usr/bin/env bash

# Example how to run:

# ./prepare_instance.sh <SSH_KEY_LOCATION> ec2-34-251-236-206.eu-west-1.compute.amazonaws.com pytorch 0.1 SQuAD2 orig ~/PycharmProjects/MemoryNet


# ./prepare_instance.sh <SSH_KEY_LOCATION> ec2-34-251-236-206.eu-west-1.compute.amazonaws.com TF 0.1 SQuAD2 orig ~/PycharmProjects/MemoryNet
# ./prepare_instance.sh <SSH_KEY_LOCATION> ec2-34-251-236-206.eu-west-1.compute.amazonaws.com pytorch 0.1 cnn-dailymail abisee ~/PycharmProjects/MemoryNet

# When you get ssh-ed to the instance finish the instance prep process by running:

# ./finish_prepare_instance.sh

# ./run_experiment.sh (optional: --terminate <AWS instance ID>)


#######################

# Args: specify one of the available datasets
#   Download the data from S3 or from local computer

# Create experiment folder structure
# Upload the AIToolbox package and install it
# Upload specified project code

#######################

key_path=$1
ec2_instance_address=$2

DL_framework=$3

AIToolbox_version=$4

dataset_name=$5
preproc_dataset=$6

local_project_path=$7


if [ $DL_framework == "TF" ]; then
    py_env="tensorflow_p36"
elif [ $DL_framework == "pytorch" ]; then
    py_env="pytorch_p36"
else
    py_env="tensorflow_p36"
fi


ssh -i $key_path ec2-user@$ec2_instance_address 'mkdir ~/project ; mkdir ~/project/data ; mkdir ~/project/model_results'

scp -i $key_path ../../dist/AIToolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/project
scp -i $key_path download_data.sh  ec2-user@$ec2_instance_address:~/project
scp -i $key_path run_experiment.sh  ec2-user@$ec2_instance_address:~/project

# Stuff for pyrouge package
scp -i $key_path ../pyrouge_set_rouge_path ec2-user@$ec2_instance_address:~/project
scp -i $key_path -r ../ROUGE-1.5.5 ec2-user@$ec2_instance_address:~/project

echo "#!/usr/bin/env bash

aws configure
cd project

source activate $py_env

pip install -U boto3
pip install awscl
pip install -U numpy
pip install --ignore-installed greenlet

pip install AIToolbox-$AIToolbox_version.tar.gz

./pyrouge_set_rouge_path ~/project/ROUGE-1.5.5

sudo yum -y install perl-CPAN
#sudo perl -MCPAN -e 'install LWP::UserAgent::Cached'
#sudo perl -MCPAN -e 'install Bundle::LWP'
sudo yum install -y perl-libwww-perl
sudo perl -MCPAN -e 'install DB_File'

cd ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
cd WordNet-2.0-Exceptions
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
cd ../..


if [ $dataset_name != '' ]; then
    ./download_data.sh ~/project/data $dataset_name $preproc_dataset
fi
" > finish_prepare_instance.sh

chmod u+x finish_prepare_instance.sh
scp -i $key_path finish_prepare_instance.sh  ec2-user@$ec2_instance_address:~/.
rm finish_prepare_instance.sh

if [ ! -z "$local_project_path" ]; then
    echo Uploading project folder $local_project_path
    source $local_project_path/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh $key_path $ec2_instance_address "~/project" $local_project_path
fi

ssh -i $key_path ec2-user@$ec2_instance_address
