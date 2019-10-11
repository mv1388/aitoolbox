import unittest
import os
import shutil
import zipfile

from aitoolbox.utils import file_system

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestFileSystem(unittest.TestCase):
    def test_zip_folder(self):
        dummy_dir_path, dummy_files_content = self.prepare_dummy_folder()

        file_system.zip_folder(dummy_dir_path, dummy_dir_path)
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
