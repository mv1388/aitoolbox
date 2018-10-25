#!/usr/bin/env bash

download_path=$1
dataset=$2
preproc_dataset=$3


aws s3 cp /Users/markovidoni/PycharmProjects/UCL_PhD_HW/data/gloveEmb/glove.6B.50d.txt.zip s3://dataset-store/glove-embeddings/glove.6B.50d.txt.zip

aws s3 cp /Users/markovidoni/PycharmProjects/UCL_PhD_HW/data/gloveEmb/glove.6B.100d.txt.zip s3://dataset-store/glove-embeddings/glove.6B.100d.txt.zip

aws s3 cp /Users/markovidoni/PycharmProjects/UCL_PhD_HW/data/gloveEmb/glove.6B.200d.txt.zip s3://dataset-store/glove-embeddings/glove.6B.200d.txt.zip

aws s3 cp /Users/markovidoni/PycharmProjects/UCL_PhD_HW/data/gloveEmb/glove.6B.300d.txt.zip s3://dataset-store/glove-embeddings/glove.6B.300d.txt.zip



function download_SQuAD2 {
    download_path=$1
    echo Downloading SQuAD2 dataset from S3
    echo "Location: $download_path"
    aws s3 cp s3://dataset-store/SQuAD2 $download_path/SQuAD2 --recursive
}

function download_cnn-dailymail {


}

if [ $dataset == "SQuAD2" ]; then
    echo Downloading SQuAD2 dataset from S3
    echo "Location: $download_path"
    aws s3 cp s3://dataset-store/SQuAD2 $download_path/SQuAD2 --recursive

elif [ $dataset == "cnn-dailymail" ]; then
    if [ -z "$preproc_dataset" ]; then
        echo Downloading original SQuAD2 dataset from S3... not yet uploaded

    elif [ $preproc_dataset == "abisee" ]; then
        echo Downloading preprocessed SQuAD2 dataset: \"abisee\"
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/cnn-dailymail/preproc/abisee $download_path/cnn-dailymail/abisee --recursive
        unzip $download_path/cnn-dailymail/abisee/cnn_stories_tokenized.zip -d $download_path/cnn-dailymail/abisee/
        unzip $download_path/cnn-dailymail/abisee/dm_stories_tokenized.zip -d $download_path/cnn-dailymail/abisee/

    elif [ $preproc_dataset == "danqi" ]; then
        echo Downloading preprocessed SQuAD2 dataset: \"danqi\"
        echo "Location: $download_path"
        aws s3 cp s3://dataset-store/cnn-dailymail/preproc/danqi $download_path/cnn-dailymail/danqi --recursive
        unzip $download_path/cnn-dailymail/danqi/cnn.tar.gz -d $download_path/cnn-dailymail/danqi/
        unzip $download_path/cnn-dailymail/danqi/dailymail.tar.gz -d $download_path/cnn-dailymail/danqi/

    else
        echo Did not find specified preprocessed dataset. Nothing will be downloaded
    fi

elif [ $dataset == "qangaroo" ]; then
    if [ -z "$preproc_dataset" ]; then
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


elif [ $dataset == "glove" ]; then
    if [ $preproc_dataset == "50" ]; then
        echo Downloading glove embeddings with dimension 50
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.50d.txt.zip $download_path/glove-embeddings/glove.6B.50d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.50d.txt.zip -d $download_path/glove-embeddings/
    elif [ $preproc_dataset == "100" ]; then
        echo Downloading glove embeddings with dimension 100
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.100d.txt.zip $download_path/glove-embeddings/glove.6B.100d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.100d.txt.zip -d $download_path/glove-embeddings/
    elif [ $preproc_dataset == "200" ]; then
        echo Downloading glove embeddings with dimension 200
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.200d.txt.zip $download_path/glove-embeddings/glove.6B.200d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.200d.txt.zip -d $download_path/glove-embeddings/
    elif [ $preproc_dataset == "300" ]; then
        echo Downloading glove embeddings with dimension 300
        aws s3 cp s3://dataset-store/glove-embeddings/glove.6B.300d.txt.zip $download_path/glove-embeddings/glove.6B.300d.txt.zip
        unzip $download_path/glove-embeddings/glove.6B.300d.txt.zip -d $download_path/glove-embeddings/
    else
        echo Specified glove embeddings dimension is not supported: 50, 100, 200, 300.
    fi

else
    echo Specified dataset not supported
fi
