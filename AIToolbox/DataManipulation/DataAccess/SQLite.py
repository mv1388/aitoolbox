import sqlite3
import os
import csv
import pickle


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
        This should keep all the data and database logic isolated to DataAccess module.
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
            sql_query (str):

        Returns:

        """
        c = self.db_conn.cursor()
        for result_row in c.execute(sql_query):
            yield result_row

    def query_db_print(self, sql_query):
        """

        Args:
            sql_query (str):

        Returns:

        """
        for row in self.query_db_generator(sql_query):
            print(row)

    def query_db_table(self, sql_query):
        """

        Args:
            sql_query (str):

        Returns:

        """
        return [row for row in self.query_db_generator(sql_query)]

    def query_db_text_save(self, sql_query, file_path, separator='\t'):
        """

        Args:
            sql_query (str):
            file_path (str):
            separator (str):

        Returns:

        """
        with open(file_path, 'w') as f:
            for row in self.query_db_generator(sql_query):
                f.write(separator.join(map(str, row)) + '\n')

    def query_db_csv_save(self, sql_query, file_path):
        """

        Args:
            sql_query (str):
            file_path (str):

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
            sql_query (str):
            dump_file_path (str):

        Returns:

        """
        pickle.dump(self.query_db_table(sql_query), open(dump_file_path, 'wb'))

    def query_db_dump_partitioned(self, sql_query, dump_file_folder_path, dump_file_name, spill_thresh=10000):
        """

        Args:
            sql_query (str):
            dump_file_folder_path (str):
            dump_file_name (str):
            spill_thresh (str):

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
            verbose (bool):

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


class SQLiteDataAccessorORM:
    """
    Implement the operations on the SQLite DB using the sqlalcehmy instead of writing the sql queries
    """
    
    def __init__(self, db_path, db_name):
        self.db_path = db_path
        self.db_name = db_name

    def query_db_generator(self, sql_query):
        raise NotImplementedError

    def persist_data_to_db(self, buffer_data, buffer_id=None, verbose=False):
        raise NotImplementedError
