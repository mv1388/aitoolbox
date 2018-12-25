import sqlite3
import mysql.connector
from pymongo import MongoClient
import os
import csv
import pickle
import networkx as nx
import codecs
import json

# try:
#     import snap
# except ImportError:
#     print 'SNAP not installed'


"""

Classes and procedures for extracting data form the database or disk based files.

"""


class SQLiteDataAccessor:
    """
    SQLite access class
    For now can: query data and :
            - print to console
            - return 2D list - table
            - save data to file on disk in text format with specified delimiter
            - save to disk in CSV
            - save to disk the 2D table in one part (potential for memory problem if query return too big)
            - save to dist the 2d table in partitions (to solve the memory problem).
                Flush to disk when enough data has accumulated.
    """

    def __init__(self, db_path='/Volumes/Zunanji Disk/MSc_StackOverflow_dump/', db_name='so-dump.db',
                 db_create_query=None, db_insert_query=None,
                 data_aggregator=None):
        """
        Examples:
            dump_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/STATS_db'
            dump_database_name = 'stats-dump.db'
            data_accessor = SQLiteDataAccessor(db_path=dump_path, db_name=dump_database_name)

            When inserting the data into the database:
                db_create_query = 'CREATE TABLE IF NOT EXISTS [SomeTable](feat1 INTEGER, feat2 TEXT)'
                db_insert_query = 'INSERT INTO SomeTable (feat1, feat2) VALUES ({values})'

            Using the data aggregator (example for TextDataStatsAggregator):
                # NLP stats calculation aggregator setup
                output_file_name = 'user_text_stats_TEST.p'
                stats_file_data_saver = FileDataAccessor(data_folder_path=output_database_path,
                                                         data_file_name=output_file_name,
                                                         data_type='pickle-dict')
                stats_aggregator = TextDataStatsAggregator(stats_file_data_saver)
                db_data_saver = SQLiteDataAccessor(db_path=output_database_path, db_name=output_database_name,
                                                   db_create_query=db_create_query, db_insert_query=db_insert_query,
                                                   data_aggregator=stats_aggregator)


        Args:
            db_path (str): Path to the folder where the DB is located. Most likely no '/' is needed at the end
            db_name (str): File name of an actual database
            db_create_query (str): Optional - Full, complete query string to be used to create a new table to which
                the data is going to be writen
            db_insert_query (str): Optional - Query string which has the same table name and the same column names to
                which the code is insering as are specified in the db_create_query query sting. The only allowed
                variable part of the string is for values - {values}.
            data_aggregator: Aggregator object that runs when all the data for current id (current user) is collected.
                Intended mostly to work as text aggregator - when all the text data from some user is collected, it
                calculates statistics on the aggregate.
        """
        self.db_path = db_path
        self.db_name = db_name
        self.db_conn = sqlite3.connect(os.path.join(self.db_path, self.db_name))

        self.delimiter = '\n'
        self.db_create_query = db_create_query
        self.db_insert_query = db_insert_query
        self.append_current_file_id = None
        self.append_current_file_data_buffer = []
        # Create a table to which we will be writing data
        if db_create_query is not None:
            self.db_conn.execute(self.db_create_query)

        # For data aggregation
        self.data_aggregator = data_aggregator

    def __del__(self):
        """
        Check if this could be used to automatically commit after finished writing to the DB.
        So no need to do it specifically in some non-data related code.
        This should keep all the data and database logic isolated to data_access module.
        """
        if self.db_create_query is not None:
            # Flush to database the last content buffer
            if len(self.append_current_file_data_buffer) > 0:
                self.persist_data_to_db([self.append_current_file_id,
                                         self.delimiter.join(self.append_current_file_data_buffer)])

                # If we have an aggregator, run it on the full combined data for one certain user
                # The idea is to have a text aggregator which produces text statistics for a certain user
                if self.data_aggregator is not None:
                    self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

            self.db_conn.commit()

        self.close_connection()

    def close_connection(self):
        """

        Returns:

        """
        self.db_conn.close()

    def execute_query(self, sql_query):
        """

        Args:
            sql_query (str):

        Returns:

        """
        c = self.db_conn.cursor()
        c.execute(sql_query)
        self.db_conn.commit()

    def query_db_generator(self, sql_query):
        """

        Args:
            sql_query:

        Returns:

        """
        c = self.db_conn.cursor()
        for result_row in c.execute(sql_query):
            yield result_row

    def query_db_print(self, sql_query):
        """

        Args:
            sql_query:

        Returns:

        """
        for row in self.query_db_generator(sql_query):
            print(row)

    def query_db_table(self, sql_query):
        """

        Args:
            sql_query:

        Returns:

        """
        return [row for row in self.query_db_generator(sql_query)]

    def query_db_text_save(self, sql_query, file_path, separator='\t'):
        """

        Args:
            sql_query:
            file_path:
            separator:

        Returns:

        """
        with open(file_path, 'w') as f:
            for row in self.query_db_generator(sql_query):
                f.write(separator.join(map(str, row)) + '\n')

    def query_db_csv_save(self, sql_query, file_path):
        """

        Args:
            sql_query:
            file_path:

        Returns:

        """

        """
            Check this: do I use wb for csv or just w
        """
        with open(file_path, 'wb') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            for row in self.query_db_generator(sql_query):
                wr.writerow(row)

    def query_db_dump(self, sql_query, dump_file_path):
        """

        Args:
            sql_query:
            dump_file_path:

        Returns:

        """
        pickle.dump(self.query_db_table(sql_query), open(dump_file_path, 'wb'))

    def query_db_dump_partitioned(self, sql_query, dump_file_folder_path, dump_file_name, spill_thresh=10000):
        """

        Args:
            sql_query:
            dump_file_folder_path:
            dump_file_name:
            spill_thresh:

        Returns:

        """
        partition_el_ctr = 0
        partition_data = []
        spill_ctr = 0

        for row in self.query_db_generator(sql_query):
            partition_data.append(row)

            partition_el_ctr += 1
            if partition_el_ctr == spill_thresh:
                pickle.dump(partition_data,
                            open(os.path.join(dump_file_folder_path, dump_file_name, '_', str(spill_ctr), '.p'), 'wb'))
                partition_data = []
                partition_el_ctr = 0
                spill_ctr += 1

        # Dump the last remaining part of data which didn't fill the partition
        if len(partition_data) > 0:
            pickle.dump(partition_data,
                        open(os.path.join(dump_file_folder_path, dump_file_name, '_', str(spill_ctr), '.p'), 'wb'))

    def save_data_to_db(self, data):
        """
        Possible duplication with persist_data_to_db() method which is already implemented below.

        Check it if I really need to implement this one as well!

        Args:
            data:

        Returns:

        """
        raise NotImplementedError

    def ordered_append_save(self, file_id, data, delimiter='\n', verbose=False):
        """

        Args:
            file_id (int): In this case file means row in teh SQLite table. Each row in the table is one user.
            data (str): In this initial implementation, data is meant to be a str clean text coming from the user's post
            delimiter (str):
            verbose:

        Returns:

        """
        self.delimiter = delimiter

        # We encountered a new user
        if file_id != self.append_current_file_id:
            # Persist the data from the previous user to the DB, before the buffer is cleared for new user
            # Check if we aren't at the first user
            if self.append_current_file_id is not None:
                self.persist_data_to_db([self.append_current_file_id,
                                         delimiter.join(self.append_current_file_data_buffer)],
                                        verbose=verbose)

                # If we have an aggregator, run it on the full combined data for one certain user
                # The idea is to have a text aggregator which produces text statistics for a certain user
                if self.data_aggregator is not None:
                    self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

            if verbose:
                print('=========  New table row added. UserId: ' + str(file_id))

            # Clean data buffer from previous user and prepare to start filling it for the new user
            self.append_current_file_data_buffer = []
            self.append_current_file_id = file_id

        self.append_current_file_data_buffer.append(data)

    def persist_data_to_db(self, buffer_data, buffer_id=None, verbose=False):
        """

        Args:
            buffer_data (list):
            buffer_id (None): In this function, this parameter is useless. Present here just for compatibility with
                FileDataAccessor's persist buffer function
            verbose (bool):

        """
        insert_query = self.db_insert_query.format(values=('?, ' * len(buffer_data))[:-2])
        self.db_conn.execute(insert_query, buffer_data)

        if verbose:
            print('Inserted data into database.')


class MySQLDataAccessor(SQLiteDataAccessor):
    def __init__(self, host, database, user, password,
                 **kwargs):
        """

        Args:
            host:
            database:
            user:
            password:
            **kwargs:
        """
        SQLiteDataAccessor.__init__(self, **kwargs)

        self.db_conn = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self.cursor = None

    def __del__(self):
        if self.cursor is not None:
            self.cursor.close()

        self.db_conn.close()

    def query_db_generator(self, sql_query):
        """

        Args:
            sql_query (str):

        Returns:

        """
        self.cursor = self.db_conn.cursor()
        self.cursor.execute(sql_query)
        for result_row in self.cursor:
            yield result_row


class MongoDBDataAccessor(SQLiteDataAccessor):
    """
        MongoDB data access. Decide if it really needs to follow the rules and inherit from the SQLiteDataAccessor.
        - DB is quite different, do I still want the same interface?
        - A lot of currently inherited functions might not work, try to prevent that.
    """

    def __init__(self, db_name, collection_name, batch_insert_size=1,
                 **kwargs):
        """

        Args:
            db_name:
            collection_name:
            batch_insert_size:
            **kwargs:
        """
        SQLiteDataAccessor.__init__(self, **kwargs)

        self.db_name = db_name
        self.collection_name = collection_name

        self.batch_insert_size = batch_insert_size
        if self.batch_insert_size > 1:
            self.insert_buffer = []

        self.client = MongoClient()
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def __del__(self):
        """
            Do I need to implement some kind of connection closing?

        Returns:

        """
        if len(self.insert_buffer) > 0:
            self.persist_buffer_data_to_db(clean_buffer=True)

    def query_db_generator(self, sql_query):
        """

        Args:
            sql_query:

        Returns:

        """
        raise NotImplementedError

    def query_db_dict(self, sql_query):
        """

        If MongoDB already returns the dict then delete this method.
        Otherwise, if each element gets returned separately (like rows in the table), then implement

        Args:
            sql_query:

        Returns:

        """
        raise NotImplementedError

    def batch_append_save(self, data, verbose=False):
        """

        Args:
            data:
            verbose:

        Returns:

        """
        self.insert_buffer.append(data)

        if len(self.insert_buffer) == self.batch_insert_size:
            self.persist_buffer_data_to_db(clean_buffer=True,
                                           verbose=verbose)

    def persist_data_to_db(self, buffer_data, buffer_id=None, verbose=False):
        """

        Args:
            buffer_data (list):
            buffer_id (None): In this function, this parameter is useless. Present here just for compatibility with
                FileDataAccessor's persist buffer function
            verbose (bool):

        """

        """
             CHECK: do I really need insert_MANY() here? Or is there a function (more optimized) for inserting only 1?
        """
        self.collection.insert_many(buffer_data)

        if verbose:
            print('Inserted data to DB.')

    def persist_buffer_data_to_db(self, clean_buffer=True, verbose=False):
        """

        Args:
            clean_buffer:
            verbose:

        Returns:

        """
        if len(self.insert_buffer) == 0:
            raise ValueError('Buffer is empty!')

        self.collection.insert_many(self.insert_buffer)

        if verbose:
            print("Inserted a batch of: " + str(len(self.insert_buffer)) + ' new documents.')

        if clean_buffer:
            self.insert_buffer = []
            if verbose:
                print('Cleaned the buffer.')


class FileDataAccessor:
    """

    """

    def __init__(self, data_folder_path='', data_file_name='', data_type='txt', data_type_options='\t'):
        """

        Examples:
            data_accessor = FileDataAccessor(data_folder_path='data/', data_file_name='full_graph.txt')

        Args:
            data_folder_path:
            data_file_name:
            data_type (str): Currently supported: 'txt', 'csv', 'json', 'pickle-dict'
            data_type_options:
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
            column_query_idx:

        Returns:

        """
        for row in self.query_db_generator(column_query_idx):
            print(row)

    def query_db_table(self, column_query_idx, max_element_limit=0):
        """

        Args:
            column_query_idx:
            max_element_limit:

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
            data:

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
            data:

        Returns:

        """
        # self.data_file_full_path and self.data_type_options for separator

        raise NotImplementedError

    def save_data_to_csv_file(self, data):
        """

        Args:
            data:

        Returns:

        """
        # Use self.data_file_full_path
        with open(self.data_file_full_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def save_data_to_json_file(self, data):
        """

        Args:
            data:

        Returns:

        """
        with open(self.data_file_full_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    def ordered_append_save(self, file_id, data, delimiter='\n', verbose=False):
        """

        Args:
            file_id:
            data:
            delimiter:
            verbose:

        Returns:

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
            buffer_data:
            buffer_id:
            verbose:

        Returns:

        """
        if self.data_type == 'pickle-dict':
            self.general_data_buffer[buffer_id] = buffer_data
        else:
            raise NotImplementedError


class DataAggregator:
    """
    DataAggregator is a wrapper that simulates similar data-flow as in the SQLiteDataAccessor but without text raw data
    saving.

    Similarly as in the case when aggregating with SQLiteDataAccessor, we also need to provide a separate data saver to
    the data_aggregator object. This DataAggregator object will not save data by it self.

    In SQLiteDataAccessor RAW data gets saved, the aggregation part still needs an additional, separate data saver.
    The

    """

    def __init__(self, data_aggregator):
        """
        Examples:
            Example for extracting only aggregate text statistics:

            database_path = '/Volumes/Zunanji Disk/MSc_StackOverflow_dump'
            database_name = 'so-dump.db'
            sql_query = '''SELECT OwnerUserId, Body FROM UserText_ONLY_accept_min2_1Column'''

            # NLP stats aggregator setup
            output_data_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/SO_db'
            output_file_name = 'user_text_stats_TEST_TEST.p'
            stats_file_data_saver = FileDataAccessor(data_folder_path=output_data_path, data_file_name=output_file_name,
                                                     data_type='pickle-dict')
            stats_aggregator = TextDataStatsAggregator(stats_file_data_saver)

            # Get the raw HTML data from the database
            data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
            # Use just the data aggregator
            data_aggregator = DataAggregator(data_aggregator=stats_aggregator)

            text_prep = TextDataPreparation()
            text_prep.extract_text_body(data_accessor, data_aggregator, sql_query, verbose=False)

        Args:
            data_aggregator: The aggregator should have method compute(data_id, data_list). An example of this is can be
                found in TextAnalysisToolbox: TextDataStatsAggregator class
        """
        self.data_aggregator = data_aggregator

        self.append_current_file_id = None
        self.append_current_file_data_buffer = []

    def __del__(self):
        """
        IMPORTANT!
        Might cause problems when doing HTML aggregation with BeautifulSoup.
        BeautifulSoup object would get destroyed before this __del__ function runs. Rely on explicitly calling
        flush_persist_remaining_data() instead to avoid this problem.

        With other types of aggregators this function worked normally... probably just luck, since object destruction is
        not done in any particular order.
        """
        # Flush and aggregate the last content buffer
        if len(self.append_current_file_data_buffer) > 0:
            self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

    def ordered_append_save(self, file_id, data, delimiter='\n', verbose=False):
        """
        The function works in the same way as the one in SQLiteDataAccessor with an exception that here always and
        ONLY aggregator is run. In this function there is NO saving of the data (cleaned text) to the database.
        The function only computes the statistics from the data and basically discards the data.

        The signature of the function is again the same as in the other classes to ensure abstraction and compatibility.

        Args:
            file_id:
            data:
            delimiter:
            verbose:

        Returns:

        """
        if file_id != self.append_current_file_id:
            # Check if we aren't at the first user
            if self.append_current_file_id is not None:
                # The idea is to have a text aggregator which produces text statistics for a certain user
                self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)

            if verbose:
                print('=========  New table row added. UserId: ' + str(file_id))

            # Clean data buffer from previous user and prepare to start filling it for the new user
            self.append_current_file_data_buffer = []
            self.append_current_file_id = file_id

        self.append_current_file_data_buffer.append(data)

    def flush_persist_remaining_data(self):
        """
        Explicit flushing of the remaining data and persisting it.
        There was a problem with order in which the objects are destroyed and __del__ code of this object used objects
        that were already destroyed.
        """
        if len(self.append_current_file_data_buffer) > 0:
            self.data_aggregator.compute(self.append_current_file_id, self.append_current_file_data_buffer)
            # Clear the buffer so that problematic __del__ function doesn't execute
            self.append_current_file_data_buffer = []


class GraphDataAccessor:
    def __init__(self, directed=False, asker_responder_direction=True):
        """

        Args:
            directed (bool): If the constructed graph is directed or undirected
            asker_responder_direction (bool): If set to True and graph is directed, the edge goes from asker user
                to responder user. If set to False, the directed edge goes into opposite direction - from responder
                user to asker user.
        """
        self.asker_responder_direction = asker_responder_direction
        if not directed:
            self.graph = nx.Graph()
        else:
            self.graph = nx.DiGraph()

    def build_graph_from_db(self, db_accessor, query,
                            verbose=False, granularity=100000, max_num_connections=10000000):
        """
        max_num_connections only works if verbose is True


        Examples:
            When using SQLLiteDataAccessor to access data from the database:

            my_query_full_graph = '''select QuestionUserId, AnswerUserId from QAFullGraph'''
            graph_builder = GraphDataAccessor(directed=True)
            graph_directed = graph_builder.build_graph_from_db(data_accessor, my_query_full_graph,
                                                               verbose=True, max_num_connections=-1)


            When using FileDataAccessor to access data from the disk file:

            column_query_idx = [1, 3]
            graph_builder = GraphDataAccessor(directed=True)
            graph_directed = graph_builder.build_graph_from_db(data_accessor, column_query_idx,
                                                               verbose=True, max_num_connections=-1)

        Args:
            db_accessor (SQLiteDataAccessor, FileDataAccessor):
            query (str):
            verbose (bool):
            granularity (int):
            max_num_connections (int):

        Returns: networkx.Graph

        """
        if verbose:
            print('Start building graph')
            for iteration, (asker_id, responder_id) in enumerate(db_accessor.query_db_generator(query)):
                if self.asker_responder_direction:
                    self.graph.add_edge(asker_id, responder_id)
                else:
                    self.graph.add_edge(responder_id, asker_id)

                if iteration % granularity == 0:
                    print(iteration)
                if iteration > max_num_connections > 0:
                    break
            print('Finished building graph')
        else:
            for asker_id, responder_id in db_accessor.query_db_generator(query):
                if self.asker_responder_direction:
                    self.graph.add_edge(asker_id, responder_id)
                else:
                    self.graph.add_edge(responder_id, asker_id)

        return self.graph

    def build_weighted_graph_from_db(self, db_accessor, query,
                                     verbose=False, granularity=100000, max_num_connections=10000000):
        """
        max_num_connections only works if verbose is True

        Args:
            db_accessor:
            query:
            verbose:
            granularity:
            max_num_connections:

        Returns:

        """
        if verbose:
            print('Start building graph')
            for iteration, (asker_id, responder_id, weight) in enumerate(db_accessor.query_db_generator(query)):
                self.graph.add_edge(asker_id, responder_id, weight=weight)
                if iteration % granularity == 0:
                    print(iteration)
                if iteration > max_num_connections > 0:
                    break
            print('Finished building graph')
        else:
            for asker_id, responder_id, weight in db_accessor.query_db_generator(query):
                self.graph.add_edge(asker_id, responder_id, weight=weight)
        return self.graph

    def save_graph(self, graph, db_accessor, save_query):
        """

        Args:
            graph:
            db_accessor:
            save_query:

        Returns:

        """
        raise NotImplementedError


# class SNAPGraphBuilder:
#     """
#     Make sure that SNAP library is installed in the interpreter you are using.
#     Also make sure that SNAP library import is uncommented at the top of this code file and that import doesn't throw
#     an ImportError exception.
#     """
#
#     def __init__(self, directed=False, asker_responder_direction=True):
#         """
#
#         Args:
#             directed (bool):
#             asker_responder_direction (bool):
#         """
#         self.asker_responder_direction = asker_responder_direction
#
#         if not directed:
#             self.graph = snap.TUNGraph.New()
#         else:
#             self.graph = snap.TNGraph.New()
#
#     def build_graph_from_db(self, db_accessor, query_nodes, query_edges,
#                             verbose=False, granularity=100000, max_num_connections=10000000):
#         """
#
#         Args:
#             db_accessor (SQLiteDataAccessor or FileDataAccessor):
#             query_nodes (str):
#             query_edges (str):
#             verbose (bool): not implemented yet
#             granularity (int): not implemented yet, possibly deprecated
#             max_num_connections (int): not implemented yet, possibly deprecated
#
#         Returns:
#
#         """
#         for node_id in db_accessor.query_db_generator(query_nodes):
#             self.graph.AddNode(int(node_id[0]))
#
#         for asker_id, responder_id in db_accessor.query_db_generator(query_edges):
#             if self.asker_responder_direction:
#                 self.graph.AddEdge(int(asker_id), int(responder_id))
#             else:
#                 self.graph.AddEdge(int(responder_id), int(asker_id))
#
#         return self.graph


class PajekDataAccessor:
    """
    Still need to decide what the functionality is going to be... basically some code is needed to prepare the dataset
    that can be used with Pajek.

    Probably need:
        load from SQLite DB and create Pajek input file
        load from file on disk and create Pajek input
    """

    def __init__(self):
        raise NotImplementedError


def get_user_reputation_lookup(database_path, database_name, sql_query,
                               output_file_path=None,
                               verbose=False):
    """
    Examples:
        database_path = '/Users/markovidoni/PycharmProjects/UniPostgrad/MSc_project/data/STATS_db'
        database_name = 'stats-dump.db'
        output_file_path = database_path + '/usr_rep_lookup.p'
        sql_query = '''SELECT Id, Reputation FROM Users'''

        get_user_reputation_lookup(database_path, database_name, sql_query, output_file_path, verbose=True)
        user_reputation = get_user_reputation_lookup(database_path, database_name, sql_query,
                                                     output_file_path=None, verbose=True)


    Args:
        database_path (str):
        database_name (str):
        sql_query (str):
        output_file_path (str, None): If str path present, pickle saves the dict to disk. If set to None, returns
            constructed dict from the function.
        verbose (bool):

    Returns:

    """

    data_accessor = SQLiteDataAccessor(db_path=database_path, db_name=database_name)
    user_reputation = {}

    for usr_id, rep in data_accessor.query_db_generator(sql_query):
        if usr_id not in user_reputation:
            user_reputation[usr_id] = rep
        else:
            print("Duplicate user")

    data_accessor.close_connection()

    if verbose:
        print("Done! Number of users: " + str(len(user_reputation)))

    if output_file_path is not None:
        if verbose:
            print('Saving to disk')
        pickle.dump(user_reputation, open(output_file_path, 'wb'))
    else:
        if verbose:
            print('Returning dict')
        return user_reputation
