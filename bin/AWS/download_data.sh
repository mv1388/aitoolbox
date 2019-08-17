#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $download_data [--dest_path STR] [--dataset STR] [--preproc STR]

   optional arguments:
     -p, --path_dest STR    destination path where to download the dataset
     -d, --dataset STR      dataset to be optionally downloaded from the S3 storage directly to ec2 instance
     -r, --preproc STR      the preprocessed version of the main dataset
     -h, --help             show this help message and exit

HEREDOC
}

# Data folder: ~/project/data (most likely)
download_path="$HOME/project/data"
dataset_name=
preproc_dataset=

while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -p|--path_dest)
    download_path="$2"
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

if [ "$download_path" == "" ] || [ "$dataset_name" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


function download_SQuAD2 {
    local download_path=$1

    echo Downloading SQuAD2 dataset from S3
    echo "Location: $download_path"
    aws s3 cp s3://dataset-store/SQuAD2 $download_path/SQuAD2 --recursive
}

function download_cnn_dailymail {
    local download_path=$1
    local preproc_dataset=$2

    if [ $preproc_dataset == "orig" ]; then
        echo Downloading original CNN-DailyMail dataset from S3... not yet uploaded

    elif [ $preproc_dataset == "abisee" ]; then
        echo Downloading preprocessed SQuAD2 dataset: \"abisee\"
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/cnn-dailymail/preproc/abisee $download_path/cnn-dailymail-abisee --recursive
        unzip $download_path/cnn-dailymail-abisee/cnn_stories_tokenized.zip -d $download_path/cnn-dailymail-abisee/
        unzip $download_path/cnn-dailymail-abisee/dm_stories_tokenized.zip -d $download_path/cnn-dailymail-abisee/

    elif [ $preproc_dataset == "danqi" ]; then
        echo Downloading preprocessed SQuAD2 dataset: \"danqi\"
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/cnn-dailymail/preproc/danqi $download_path/cnn-dailymail-danqi --recursive
        tar xvf $download_path/cnn-dailymail-danqi/cnn.tar.gz -C $download_path/cnn-dailymail-danqi/
        tar xvf $download_path/cnn-dailymail-danqi/dailymail.tar.gz -C $download_path/cnn-dailymail-danqi/

    else
        echo Did not find specified preprocessed dataset. Nothing will be downloaded
    fi
}

function download_qangaroo {
    local download_path=$1
    local preproc_dataset=$2

    if [ $preproc_dataset == "orig" ]; then
        echo Downloading both original qangaroo datsets: medhop and wikihop
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/qangaroo_v1 $download_path/qangaroo_v1 --recursive
        unzip $download_path/qangaroo_v1/medhop.zip -d $download_path/qangaroo_v1/
        unzip $download_path/qangaroo_v1/wikihop.zip -d $download_path/qangaroo_v1/

    elif [ $preproc_dataset == "medhop-orig" ]; then
        echo Downloading only original qangaroo medhop dataset
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/qangaroo_v1/medhop.zip $download_path/qangaroo_v1/medhop.zip
        unzip $download_path/qangaroo_v1/medhop.zip -d $download_path/qangaroo_v1/
    elif [ $preproc_dataset == "wikihop-orig" ]; then
        echo Downloading only original qangaroo wikihop dataset
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/qangaroo_v1/wikihop.zip $download_path/qangaroo_v1/wikihop.zip
        unzip $download_path/qangaroo_v1/wikihop.zip -d $download_path/qangaroo_v1/
    else
        echo Did not find specified preprocessed dataset. Nothing will be downloaded
    fi
}

function download_HotpotQA {
    local download_path=$1
#    local preproc_dataset=$2

    echo Downloading HotpotQA dataset from S3
    echo "Location: $download_path"
    aws s3 cp s3://dataset-store/HotpotQA $download_path/HotpotQA --recursive
    unzip $download_path/HotpotQA/HotpotQA.zip -d $download_path/HotpotQA/
}

function download_glove {
    local download_path=$1
    local preproc_dataset=$2

    if [ $preproc_dataset == "50" ]; then
        echo Downloading glove embeddings with dimension 50
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.50d.txt.zip $download_path/glove-embeddings/glove.6B.50d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.50d.txt.zip -d $download_path/glove-embeddings/
        rm -r $download_path/glove-embeddings/__MACOSX
    elif [ $preproc_dataset == "100" ]; then
        echo Downloading glove embeddings with dimension 100
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.100d.txt.zip $download_path/glove-embeddings/glove.6B.100d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.100d.txt.zip -d $download_path/glove-embeddings/
        rm -r $download_path/glove-embeddings/__MACOSX
    elif [ $preproc_dataset == "200" ]; then
        echo Downloading glove embeddings with dimension 200
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.200d.txt.zip $download_path/glove-embeddings/glove.6B.200d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.200d.txt.zip -d $download_path/glove-embeddings/
        rm -r $download_path/glove-embeddings/__MACOSX
    elif [ $preproc_dataset == "300" ]; then
        echo Downloading glove embeddings with dimension 300
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.300d.txt.zip $download_path/glove-embeddings/glove.6B.300d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.300d.txt.zip -d $download_path/glove-embeddings/
        rm -r $download_path/glove-embeddings/__MACOSX
    else
        echo Specified glove embeddings dimension is not supported: 50, 100, 200, 300.
    fi
}


if [ $dataset_name == "SQuAD2" ]; then
    download_SQuAD2 $download_path

elif [ $dataset_name == "cnn-dailymail" ]; then
    download_cnn_dailymail $download_path $preproc_dataset

elif [ $dataset_name == "qangaroo" ]; then
    download_qangaroo $download_path $preproc_dataset

elif [ $dataset_name == "HotpotQA" ]; then
    download_HotpotQA $download_path

elif [ $dataset_name == "glove" ]; then
    download_glove $download_path $preproc_dataset

else
    echo Specified dataset not supported
fi
