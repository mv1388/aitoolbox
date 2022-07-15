import os
import subprocess
import time
import datetime
import pandas as pd

import typer


class TrainingJobScheduler:
    def __init__(self, job_queue_file_path):
        self.job_queue_file_path = os.path.expanduser(job_queue_file_path)
        self.job_queue = None

        self.job_counter = 0

    def run_jobs(self, logging_path, log_s3_dir_path, aws_region):
        self.job_queue = pd.read_csv(self.job_queue_file_path)

        while self.is_job_available():
            if len(self.job_queue[self.job_queue['job_status'] == 'running']):
                raise ValueError

            jobs_waiting = self.job_queue[self.job_queue['job_status'] == 'waiting']
            job_todo = jobs_waiting.head(1)

            self.job_queue.loc[job_todo.index, 'job_status'] = 'running'
            self.job_queue.to_csv(self.job_queue_file_path, index=False)

            logging_path = self.get_job_logging_path(logging_path)
            log_upload_setting = f"--log-path {logging_path} --log-s3-upload-dir {log_s3_dir_path}"

            process_return = subprocess.run(
                f"{os.path.expanduser('~/project/run_experiment.sh')} "
                f"--experiment-script {job_todo.iloc[0]['experiment_script_file']} "
                f"--project-root {job_todo.iloc[0]['project_root_path']} "
                f"{log_upload_setting} "
                f"--cleanup-script "
                f"--aws-region {aws_region}",
                shell=True
            )

            # re-read the queue file to get in any additions to the queue during the model training run
            self.job_queue = pd.read_csv(self.job_queue_file_path)
            self.job_queue.loc[job_todo.index, 'job_status'] = 'done'
            self.job_queue.loc[job_todo.index, 'job_return_code'] = process_return.returncode
            self.job_queue.to_csv(self.job_queue_file_path, index=False)

            self.job_counter += 1

    def is_job_available(self):
        self.job_queue = pd.read_csv(self.job_queue_file_path)
        return not all(el == 'done' for el in self.job_queue['job_status'])

    def get_job_logging_path(self, logging_path):
        path_extension = os.path.expanduser(logging_path).split('.')
        if len(path_extension) != 2:
            raise ValueError

        logging_path, extension = path_extension

        return f'{logging_path}_{self.job_counter}.{extension}'

    def add_job(self, experiment_script_file, project_root_path):
        if os.path.exists(self.job_queue_file_path):
            self.job_queue = pd.read_csv(self.job_queue_file_path)
        else:
            self.job_queue = pd.DataFrame(columns=['job_status', 'experiment_script_file',
                                                   'project_root_path', 'job_return_code', 'timestamp'])

        self.job_queue = self.job_queue.append({
            'job_status': 'waiting',
            'experiment_script_file': experiment_script_file,
            'project_root_path': project_root_path,
            'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        }, ignore_index=True)

        self.job_queue.to_csv(self.job_queue_file_path, index=False)

    def __str__(self):
        return str(pd.read_csv(self.job_queue_file_path))


app = typer.Typer(help='Training Job Scheduler CLI')


@app.command(help='Run training jobs execution loop which goes runs through provided jobs in the queue')
def run(
        log_path: str = typer.Option(
            os.path.expanduser(f"~/project/training_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H_%M_%S')}.log"),
            help='Logging file path on the execution server'
        ),
        log_s3_upload_dir: str = typer.Option(
            's3://model-result/training_logs',
            help='Path to the logs folder on S3 to which the training log should be uploaded'
        ),
        aws_region: str = typer.Option(
            'eu-west-1',
            help='AWS region code'
        ),
        terminate: bool = typer.Option(
            False,
            help='The instance will be terminated when all the training is done'
        ),
        job_queue_file_path: str = typer.Option(
            '~/training_job_queue.csv',
            help='File path of the job queue on the execution server/AWS'
        )
):
    job_scheduler = TrainingJobScheduler(job_queue_file_path)
    print('Jobs currently in the queue:')
    print(job_scheduler)

    job_scheduler.run_jobs(log_path, log_s3_upload_dir, aws_region)

    if terminate:
        print('Terminating the instance')
        subprocess.run(
            'aws ec2 terminate-instances --instance-ids $(ec2metadata --instance-id | cut -d " " -f 2)',
            shell=True
        )


@app.command(help='Add a new training job to the job queue')
def add_job(
        experiment_script: str = typer.Option(
            'aws_run_experiments_project.sh',
            help='Name of the experiment bash script to be executed in order to start the training'
        ),
        project_root: str = typer.Option(
            '~/project',
            help='Path to the project root on the execution server/AWS'
        ),
        job_queue_file_path: str = typer.Option(
            '~/training_job_queue.csv',
            help='File path of the job queue on the execution server/AWS'
        )
):
    job_scheduler = TrainingJobScheduler(job_queue_file_path)
    job_scheduler.add_job(experiment_script, project_root)

    print('Job added!')
    print(job_scheduler)


@app.command(help='List the job queue contents')
def list_queue(
        job_queue_file_path: str = typer.Option(
            '~/training_job_queue.csv',
            help='File path of the job queue on the execution server/AWS'
        )
):
    print(TrainingJobScheduler(job_queue_file_path))


if __name__ == "__main__":
    app()
