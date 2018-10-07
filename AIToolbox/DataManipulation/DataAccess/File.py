import os
import csv
import pickle
import codecs
import json


class FileDataAccessor:
    def __init__(self, data_folder_path='', data_file_name='', data_type='txt', data_type_options='\t'):
        """

        Examples:
            data_accessor = FileDataAccessor(data_folder_path='data/', data_file_name='full_graph.txt')

        Args:
            data_folder_path (str):
            data_file_name (str):
            data_type (str): Currently supported: 'txt', 'csv', 'json', 'pickle-dict'
            data_type_options (str):
        """
        self.data_folder_path = data_folder_path
        self.data_file_name = data_file_name
        self.data_file_full_path = os.path.join(self.data_folder_path, self.data_file_name)

        self.data_type = data_type
        self.data_type_options = data_type_options

        # For ordered_append_save()
        self.append_current_file_id = None
        self.append_current_file = None

        # For persist_data_to_db()
        self.general_data_buffer = None
        if data_type == 'pickle-dict':
            self.general_data_buffer = {}

    def __del__(self):
        """
        Closes last file in the folder append procedure after we stopped adding new content

        Saves the general data buffer to disk
        """
        if self.append_current_file is not None:
            self.append_current_file.close()

        if self.general_data_buffer is not None and len(self.general_data_buffer) > 0:
            # A better option when other pickle saving data structures are added:
            # if self.data_type[:6] == 'pickle':
            if self.data_type == 'pickle-dict':
                pickle.dump(self.general_data_buffer, open(self.data_file_full_path, 'wb'))

    def query_db_generator(self, column_query_idx):
        """
        IMPORTANT: must be named query_db_generator

        if data type is TXT, self.data_type_options must specify the separator used in the saved data.

        Args:
            column_query_idx (list): set to None to get all the columns in the file

        Returns:

        """
        if self.data_type == 'txt':
            with open(self.data_file_full_path, 'r') as f:
                for row in f:
                    row_list = row.strip().split(self.data_type_options)
                    if column_query_idx is None:
                        yield row_list
                    else:
                        yield [row_list[idx] for idx in column_query_idx]
        elif self.data_type == 'csv':
            with open(self.data_file_full_path, 'rb') as f:
                reader = csv.reader(f)
                for row_list in reader:
                    if column_query_idx is None:
                        yield row_list
                    else:
                        yield [row_list[idx] for idx in column_query_idx]
        elif self.data_type == 'json':
            with open(self.data_file_full_path) as json_file:
                json_data_list = json.load(json_file)
                for el in json_data_list:
                    if column_query_idx is None:
                        yield el
                    else:
                        if len(column_query_idx) == 1:
                            yield el[column_query_idx[0]]
                        else:
                            yield {idx: el[idx] for idx in column_query_idx}
        else:
            print('Not supported data format')

    def query_db_print(self, column_query_idx):
        """

        Args:
            column_query_idx (list):

        Returns:

        """
        for row in self.query_db_generator(column_query_idx):
            print(row)

    def query_db_table(self, column_query_idx, max_element_limit=0):
        """

        Args:
            column_query_idx (list):
            max_element_limit (int):

        Returns:

        """
        if max_element_limit <= 0:
            if self.data_type == 'json' and column_query_idx is None:
                with open(self.data_file_full_path) as json_file:
                    json_data_list = json.load(json_file)
                return json_data_list
            else:
                return [row for row in self.query_db_generator(column_query_idx)]
        else:
            table = []
            for idx, row in enumerate(self.query_db_generator(column_query_idx)):
                table.append(row)
                if idx == max_element_limit:
                    break
            return table

    def save_data_to_file(self, data):
        """
        Interface method

        Args:
            data (list):

        Returns:

        """
        if self.data_type == 'txt':
            self.save_data_to_text_file(data)
        elif self.data_type == 'csv':
            self.save_data_to_csv_file(data)
        elif self.data_type == 'json':
            self.save_data_to_json_file(data)
        else:
            print('Not supported data format')

    def save_data_to_text_file(self, data):
        """

        Args:
            data (list):

        Returns:

        """
        # self.data_file_full_path and self.data_type_options for separator

        raise NotImplementedError

    def save_data_to_csv_file(self, data):
        """

        Args:
            data (list):

        Returns:

        """
        # Use self.data_file_full_path
        with open(self.data_file_full_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def save_data_to_json_file(self, data):
        """

        Args:
            data (list):

        Returns:

        """
        with open(self.data_file_full_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    def ordered_append_save(self, file_id, data, delimiter='\n', verbose=False):
        """

        Args:
            file_id (int):
            data (list):
            delimiter (str):
            verbose (bool):

        Returns:
            None

        """
        if file_id != self.append_current_file_id:
            if verbose:
                print('=========  New file opened. FileId: ' + str(file_id))

            # Close previous file
            if self.append_current_file is not None:
                self.append_current_file.close()
            # Open new file for new user for example
            # self.append_current_file = open(os.path.join(self.data_folder_path, str(file_id) + '.' + self.data_type),
            #                                 'a')
            # Use to be able to write non ASCI characters to file - write in utf-8
            self.append_current_file = codecs.open(os.path.join(self.data_folder_path,
                                                                str(file_id) + '.' + self.data_type),
                                                   'a', encoding='utf-8')
            self.append_current_file_id = file_id

        if verbose:
            print('Appending to file: ' + str(file_id))

        self.append_current_file.write(data + delimiter)

    def persist_data_to_db(self, buffer_data, buffer_id=None, verbose=False):
        """
        This is a general interface method for persisting data to disk.

        Currently only saving to pickle in a form of a dict is supported. Extend if needed (pickle as a table,
        text file table, csv)

        NOTICE: Naming of the function: even though it says to_db it is actually not really saving to the database.
                Such name was only used to be compatible with the corresponding function in the SQLiteDataAccessor.
                This way a common, data saver independent interface can be used by the code using these savers.

        Args:
            buffer_data (list):
            buffer_id (int):
            verbose (bool):

        Returns:
            None

        """
        if self.data_type == 'pickle-dict':
            self.general_data_buffer[buffer_id] = buffer_data
        else:
            raise NotImplementedError
