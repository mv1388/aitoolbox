import csv
from pymongo import MongoClient


class MongoDBDataHandler:
    def __init__(self, db_name_source, collection_name_source):
        self.db_name = db_name_source
        self.collection_name = collection_name_source

    def remove_collection(self):
        raise NotImplementedError

    def copy_collection(self, db_name_destination, collection_name_destination):
        raise NotImplementedError

    def add_to_collection(self, db_name_destination, collection_name_destination):
        raise NotImplementedError

    def combie_collection_list(self):
        assert isinstance(self.collection_name, list)
        raise NotImplementedError

    def move_collection(self, db_name_destination, collection_name_destination):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError



class RPyDataHandler:
    def __init__(self, data=None):
        self.data = data

    # Data reader fns
    def read_R_data_frame(self, path):
        read_data = None # Find fn to read data from .rda which contains data frame

        if self.data is not None:
            print "Data already present in Data Handler. Just returning data and not saving it."
            return read_data
        elif self.data is None:
            self.data = read_data
            return read_data


    # Data saver fns
    def save_to_R_data_frame(self, path):
        # Save file to .rda file at path

        return None


class DataReader:
    def __init__(self):
        pass


class DataWriter:
    def __init__(self, data, output_file_folder, output_file_name):
        self.data = data
        self.output_file_folder = output_file_folder
        self.output_file_name = output_file_name

    def csv_write(self, file_ext='.csv'):
        with open(self.output_file_folder + self.output_file_name + file_ext, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for r in self.data:
                writer.writerow(r)
