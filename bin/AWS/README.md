# AWS scripts

## Instance preparation

Main script for preparing the instance after it has been created via web console: 
[prepare_instance.sh](prepare_instance.sh)

```bash
./prepare_instance.sh <SSH_KEY_LOCATION> <AWS_INSTANCE_ADDRESS> 0.1 SQuAD2 orig ~/PycharmProjects/MemoryNet
```

`prepare_instance` script at the end ssh connects to the instance. 
There `finish_prepare_instance.sh` is already present. To finish the instance preparation execute on the instance:

```bash
./finish_prepare_instance.sh
```


## Run experiments

When ssh-ed to the instance the [run_experiment.sh](run_experiment.sh) script is found in the
`~/project` folder on the instance.
Execute this script to run the experiments specified in the main project's `aws_run_experiments_project.sh` script.

```bash
./run_experiment.sh 
```

To automatically terminate the instance after the training is done execute with `--terminate` option:

```bash
./run_experiment.sh --terminate <AWS_INSTANCE_ID>
```


## Download dataset from S3

```bash
./download_data.sh ~/PycharmProjects/MemoryNet/data <dataset_name> <preproc_dataset>
```
