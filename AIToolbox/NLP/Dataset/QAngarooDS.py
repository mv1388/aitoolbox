from AIToolbox.AWS.DataAccess import QAngarooDSDatasetFetcher


def get_dataset_local_copy(local_dataset_folder_path, protect_local_folder=True):
    """Interface method for getting a local copy of QAngaroo dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        protect_local_folder (bool):

    Returns:
        None

    """
    dataset_fetcher = QAngarooDSDatasetFetcher(bucket_name='dataset-store', local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_dataset(protect_local_folder)
