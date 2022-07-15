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

            subprocess.run(
                [
                    os.path.expanduser('~/project/run_experiment.sh'),
                    '--experiment-script', job_todo['experiment_script_file'],
                    '--project-root', job_todo['project_root_path'],
                    log_upload_setting,
                    '--cleanup-script',
                    '--aws-region', aws_region
                ]
            )

            self.job_queue.loc[job_todo.index, 'job_status'] = 'done'
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
                                                   'project_root_path', 'timestamp'])

        self.job_queue = self.job_queue.append({
            'job_status': 'waiting',
            'experiment_script_file': experiment_script_file,
            'project_root_path': project_root_path,
            'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        }, ignore_index=True)

        self.job_queue.to_csv(self.job_queue_file_path, index=False)

    def __str__(self):
        return str(pd.read_csv(self.job_queue_file_path))


app = typer.Typer()


@app.command()
def run(
        log_path: str, log_s3_upload_dir: str, aws_region: str,
        terminate: bool = False,
        job_queue_file_path: str = '~/training_job_queue.csv'
):
    job_scheduler = TrainingJobScheduler(job_queue_file_path)
    print('Jobs currently in the queue:')
    print(job_scheduler)

    job_scheduler.run_jobs(log_path, log_s3_upload_dir, aws_region)

    if terminate:
        print('Terminating the instance')
        subprocess.run('aws ec2 terminate-instances --instance-ids $(ec2metadata --instance-id | cut -d " " -f 2)')


@app.command()
def add_job(experiment_script: str = 'aws_run_experiments_project.sh', project_root: str = '~/project',
            job_queue_file_path: str = '~/training_job_queue.csv'):
    job_scheduler = TrainingJobScheduler(job_queue_file_path)
    job_scheduler.add_job(experiment_script, project_root)

    print('Job added!')
    print(job_scheduler)


@app.command()
def list_queue(job_queue_file_path: str = '~/training_job_queue.csv'):
    print(TrainingJobScheduler(job_queue_file_path))


if __name__ == "__main__":
    app()
