import os
from os import path


def create_folder_hierarchy(base_folder_path, folder_names):
    """

    Args:
        base_folder_path (str):
        folder_names (list):

    Returns:
        str:
    """
    if not path.exists(base_folder_path):
        raise ValueError(f'Provided base folder does not exist: {base_folder_path}')

    folder_path = base_folder_path

    for folder_name in folder_names:
        folder_path = path.join(folder_path, folder_name)

        if not path.exists(folder_path):
            os.mkdir(folder_path)

    return folder_path
