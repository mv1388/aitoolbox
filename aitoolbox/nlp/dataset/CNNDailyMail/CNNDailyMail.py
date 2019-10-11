from aitoolbox.cloud.AWS.data_access import CNNDailyMailDatasetFetcher


def get_preproc_dataset_local_copy(local_dataset_folder_path, preprocess_name='abisee', protect_local_folder=True):
    """Interface method for getting a local copy of CNN/DailyMail dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        preprocess_name (str):
        protect_local_folder (bool):

    Returns:
        None

    """
    dataset_fetcher = CNNDailyMailDatasetFetcher(bucket_name='dataset-store', local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_preprocessed_dataset(preprocess_name, protect_local_folder)
