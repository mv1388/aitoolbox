from aitoolbox.cloud.AWS.data_access import HotpotQADatasetFetcher


"""

    https://hotpotqa.github.io/
    https://arxiv.org/pdf/1809.09600.pdf

    https://github.com/hotpotqa/hotpot

"""


def get_dataset_local_copy(local_dataset_folder_path, protect_local_folder=True):
    """Interface method for getting a local copy of HotpotQA dataset

    If a local copy is not found, dataset is automatically downloaded from S3.

    Args:
        local_dataset_folder_path (str):
        protect_local_folder (bool):

    Returns:
        None

    """
    dataset_fetcher = HotpotQADatasetFetcher(bucket_name='dataset-store', local_dataset_folder_path=local_dataset_folder_path)
    dataset_fetcher.fetch_dataset(protect_local_folder=protect_local_folder)
