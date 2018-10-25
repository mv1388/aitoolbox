#!/usr/bin/env bash

# Example how to run:

# ./prepare_instance.sh <SSH_KEY_LOCATION> ec2-54-194-81-48.eu-west-1.compute.amazonaws.com 0.0.7 SQuAD2

# When you get ssh-ed to the instance finish the instance prep process by running:

# ./finish_prepare_instance.sh


#######################

# Args: specify one of the available datasets
#   Download the data from S3 or from local computer

# Upload the AIToolbox package and install it
# Upload specified project code
# Create experiment folder structure

#######################

key_path=$1
ec2_instance_address=$2

AIToolbox_version=$3

dataset_name=$4
preproc_dataset=$5


scp -i $key_path ../../dist/AIToolbox-$AIToolbox_version.tar.gz  ec2-user@$ec2_instance_address:~/.
scp -i $key_path download_data.sh  ec2-user@$ec2_instance_address:~/.

echo "#!/usr/bin/env bash

sudo -H pip install awscli --upgrade
#sudo yum downgrade aws-cli.noarch python27-botocore

aws configure

mkdir project
mv AIToolbox-$AIToolbox_version.tar.gz project/
mv download_data.sh project/
cd project
mkdir model_results
mkdir data

source activate tensorflow_p36
pip install AIToolbox-$AIToolbox_version.tar.gz

if [ $dataset_name != '' ]; then
    ./download_data.sh ~/project/data $dataset_name $preproc_dataset
fi
" > finish_prepare_instance.sh

chmod u+x finish_prepare_instance.sh
scp -i $key_path finish_prepare_instance.sh  ec2-user@$ec2_instance_address:~/.

rm finish_prepare_instance.sh

ssh -i $key_path ec2-user@$ec2_instance_address
