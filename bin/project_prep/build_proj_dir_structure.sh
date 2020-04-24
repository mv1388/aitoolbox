#!/usr/bin/env bash

project_root_path=$1
project_name=$2

mkdir $project_root_path/$project_name

mkdir $project_root_path/$project_name/data
mkdir $project_root_path/$project_name/code_lib
mkdir $project_root_path/$project_name/experiments
mkdir $project_root_path/$project_name/model_results
mkdir $project_root_path/$project_name/notebooks
mkdir $project_root_path/$project_name/papers
mkdir $project_root_path/$project_name/AWS_run_scripts
mkdir $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts
mkdir $project_root_path/$project_name/tst_code

cp template_gitignore $project_root_path/$project_name/.gitignore

cp template_aws_project_upload.sh $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh
chmod u+x $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts/aws_project_upload.sh

cp template_aws_run_experiments_project.sh $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts/aws_run_experiments_project.sh
chmod u+x $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts/aws_run_experiments_project.sh

touch $project_root_path/$project_name/AWS_run_scripts/AWS_core_scripts/requirements.txt
