#!/usr/bin/env bash

dataset=$1
download_path=$2

if [ $dataset == "SQuAD2" ]; then
    echo Downloading SQuAD2 dataset from S3
    echo "Location: $download_path"
    aws s3 cp s3://dataset-store/SQuAD2 $download_path/SQuAD2 --recursive
else
    echo Specified dataset not supported
fi
