from aitoolbox.cloud.AWS.data_access import QAngarooDatasetFetcher


def get_dataset_local_copy(local_dataset_folder_path, dataset_name=None, protect_local_folder=True):
    """Interface method for getting a local copy of QAngaroo dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        dataset_name (str or None): possible options: medhop, wikihop or None
        protect_local_folder (bool):

    Returns:
        None

    """
    dataset_fetcher = QAngarooDatasetFetcher(bucket_name='dataset-store', local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_dataset(dataset_name=dataset_name, protect_local_folder=protect_local_folder)
