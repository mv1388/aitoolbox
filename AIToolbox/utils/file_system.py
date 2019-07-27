import os
from os import path


def create_folder_hierarchy(base_folder_path, folder_names):
    """

    Args:
        base_folder_path (str): folder from which the created folder hierarchy will go into further depth
        folder_names (list): names of folders to be created one inside the previous

    Returns:
        str, list: path to final folder in hierarchy, all the folder paths in the created hierarchy
    """
    if not path.exists(base_folder_path):
        raise ValueError(f'Provided base folder does not exist: {base_folder_path}')

    folder_path = base_folder_path
    all_created_folder_paths = [folder_path]

    for folder_name in folder_names:
        folder_path = path.join(folder_path, folder_name)
        all_created_folder_paths.append(folder_path)

        if not path.exists(folder_path):
            os.mkdir(folder_path)

    return folder_path, all_created_folder_paths
