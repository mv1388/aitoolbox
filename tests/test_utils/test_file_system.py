import unittest
import os
import shutil
import zipfile

from aitoolbox.utils import file_system

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestFileSystem(unittest.TestCase):
    def test_create_folder_hierarchy(self):
        folder_hierarchy = ['folder_1', 'folder_2', 'folder_3']
        folder_path, all_created_folder_paths = file_system.create_folder_hierarchy(THIS_DIR, folder_hierarchy)

        self.assertTrue(os.path.exists(os.path.join(THIS_DIR, *folder_hierarchy)))
        self.assertEqual(folder_path, os.path.join(THIS_DIR, *folder_hierarchy))
        self.assertEqual(
            all_created_folder_paths,
            [os.path.join(THIS_DIR, *folder_hierarchy[:i]) for i in range(len(folder_hierarchy) + 1)]
        )

        with self.assertRaises(ValueError):
            file_system.create_folder_hierarchy('missing_path', ['folder_1', 'folder_2'])

        if os.path.exists(os.path.join(THIS_DIR, 'folder_1')):
            shutil.rmtree(os.path.join(THIS_DIR, 'folder_1'))

    def test_create_folder_hierarchy_two_prong(self):
        folder_hierarchy_1 = ['folder_1', 'folder_2', 'folder_3']
        folder_path_1, all_created_folder_paths_1 = file_system.create_folder_hierarchy(THIS_DIR, folder_hierarchy_1)

        self.assertTrue(os.path.exists(os.path.join(THIS_DIR, *folder_hierarchy_1)))
        self.assertEqual(folder_path_1, os.path.join(THIS_DIR, *folder_hierarchy_1))
        self.assertEqual(
            all_created_folder_paths_1,
            [os.path.join(THIS_DIR, *folder_hierarchy_1[:i]) for i in range(len(folder_hierarchy_1) + 1)]
        )

        folder_hierarchy = ['folder_1', 'folder_4', 'folder_1']
        folder_path, all_created_folder_paths = file_system.create_folder_hierarchy(THIS_DIR, folder_hierarchy)

        # Check if previously created hierarchy still exists
        self.assertTrue(os.path.exists(os.path.join(THIS_DIR, *folder_hierarchy_1)))

        self.assertTrue(os.path.exists(os.path.join(THIS_DIR, *folder_hierarchy)))
        self.assertEqual(folder_path, os.path.join(THIS_DIR, *folder_hierarchy))
        self.assertEqual(
            all_created_folder_paths,
            [os.path.join(THIS_DIR, *folder_hierarchy[:i]) for i in range(len(folder_hierarchy) + 1)]
        )

        with self.assertRaises(ValueError):
            file_system.create_folder_hierarchy('missing_path', ['folder_1', 'folder_2'])

        if os.path.exists(os.path.join(THIS_DIR, 'folder_1')):
            shutil.rmtree(os.path.join(THIS_DIR, 'folder_1'))

    def test_zip_folder(self):
        self.zip_folder()

    def test_zip_folder_with_zip_extension(self):
        self.zip_folder(with_zip_extension=True)

    def zip_folder(self, with_zip_extension=False):
        dummy_dir_path, dummy_files_content = self.prepare_dummy_folder()

        if with_zip_extension:
            dummy_zip_path = f'{dummy_dir_path}.zip'
        else:
            dummy_zip_path = dummy_dir_path

        file_system.zip_folder(dummy_dir_path, dummy_zip_path)
        self.assertTrue(os.path.exists(dummy_dir_path + '.zip'))

        with zipfile.ZipFile(dummy_dir_path + '.zip', 'r') as zip_ref:
            zip_ref.extractall(dummy_dir_path + 'unzipped')

        self.assertEqual(sorted(os.listdir(dummy_dir_path + 'unzipped')),
                         sorted([f'file_{i}.txt' for i in range(len(dummy_files_content))]))

        for i, correct_text in enumerate(dummy_files_content):
            with open(os.path.join(dummy_dir_path + 'unzipped', f'file_{i}.txt'), 'r') as f:
                f_content = f.read()
                self.assertEqual(f_content, correct_text)

        if os.path.exists(dummy_dir_path):
            shutil.rmtree(dummy_dir_path)

        if os.path.exists(dummy_dir_path + 'unzipped'):
            shutil.rmtree(dummy_dir_path + 'unzipped')

        if os.path.exists(dummy_dir_path + '.zip'):
            os.remove(dummy_dir_path + '.zip')

    def test_unzip_file(self):
        dummy_dir_path, dummy_files_content = self.prepare_dummy_folder()
        file_system.zip_folder(dummy_dir_path, dummy_dir_path)
        self.assertTrue(os.path.exists(dummy_dir_path + '.zip'))

        if os.path.exists(dummy_dir_path):
            shutil.rmtree(dummy_dir_path)

        file_system.unzip_file(dummy_dir_path + '.zip', dummy_dir_path)

        self.assertEqual(sorted(os.listdir(dummy_dir_path)),
                         sorted([f'file_{i}.txt' for i in range(len(dummy_files_content))]))

        for i, correct_text in enumerate(dummy_files_content):
            with open(os.path.join(dummy_dir_path, f'file_{i}.txt'), 'r') as f:
                f_content = f.read()
                self.assertEqual(f_content, correct_text)

        if os.path.exists(dummy_dir_path):
            shutil.rmtree(dummy_dir_path)

        if os.path.exists(dummy_dir_path + '.zip'):
            os.remove(dummy_dir_path + '.zip')

    @staticmethod
    def prepare_dummy_folder():
        dummy_dir_path = os.path.join(THIS_DIR, 'dummy_dir')
        os.mkdir(dummy_dir_path)

        dummy_files_content = ['bla bla', 'kuku lele', 'how how 1234000']

        for i, text in enumerate(dummy_files_content):
            with open(os.path.join(dummy_dir_path, f'file_{i}.txt'), 'w') as f:
                f.write(text)

        return dummy_dir_path, dummy_files_content
