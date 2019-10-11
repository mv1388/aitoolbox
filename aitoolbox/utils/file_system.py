import os
from os import path
import shutil
import zipfile
import tarfile


def create_folder_hierarchy(base_folder_path, folder_names):
    """Create nested folder hierarchy

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


def zip_folder(source_dir_path, zip_path):
    """Utility function for zipping a folder into .zip archive

    Args:
        source_dir_path (str): path to the folder that is going to be zipped
        zip_path (str): specify the path of the zip file which will be created

    Returns:
        str: the full path to the produced zip file (with the .zip extension appended)
    """
    if zip_path[-4:] == '.zip':
        zip_path = zip_path[:-4]

    shutil.make_archive(zip_path, 'zip', source_dir_path)
    return zip_path + '.zip'


def unzip_file(file_path, target_dir_path):
    """Util function for zip file unzipping

    Args:
        file_path (str): path to the zip file
        target_dir_path (str): destination where unzipped content is stored
    """
    if file_path[-4:] == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir_path)
    elif file_path[-7:] == '.tar.gz':
        with tarfile.open(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir_path)
