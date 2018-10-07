from pymongo import MongoClient

from AIToolbox.DataManipulation.DataAccess.SQLite import *


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
            db_name (str):
            collection_name (str):
            batch_insert_size (int):
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
            clean_buffer (bool):
            verbose (bool):

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


class MongoDBWriter:
    def __init__(self, db_name, collection_name, batch_insert_size=1, verbose=False):
        """

        Args:
            db_name (str):
            collection_name (str):
            batch_insert_size (int):
            verbose (bool):
        """
        self.verbose = verbose

        self.db_name = db_name
        self.collection_name = collection_name

        self.batch_insert_size = batch_insert_size
        if self.batch_insert_size > 1:
            self.insert_buffer = []

        self.client = MongoClient()
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    # Todo: new insert consistency checking code
    def check_consistency(self):
        pass

    def batch_insert(self, document):
        """

        Args:
            document:

        Returns:

        """
        self.insert_buffer.append(document)

        if len(self.insert_buffer) == self.batch_insert_size:
            self.collection.insert_many(self.insert_buffer)

            self.insert_buffer = []

            if self.verbose:
                print("Inserted a batch of: " + str(self.batch_insert_size) + ' new documents.')


class MongoDBMultiTopicWriter:
    def __init__(self, db_name, collections_topics, unclaimed_collection, batch_insert_size=1, verbose=False):
        """

        Args:
            db_name (str):
            collections_topics:
            unclaimed_collection:
            batch_insert_size (int):
            verbose (bool):
        """
        self.verbose = verbose

        self.db_name = db_name
        self.collections_topics = collections_topics

        self.batch_insert_size = batch_insert_size
        self.topic_insert_buffer = {}

        self.client = MongoClient()
        self.db = self.client[self.db_name]
        self.collection = {}

        for topic_collection_name in self.collections_topics:
            self.topic_insert_buffer[topic_collection_name] = []
            self.collection[topic_collection_name] = self.db[topic_collection_name]

        # Handle unclaimed tweets collection
        self.topic_insert_buffer[unclaimed_collection] = []
        self.collection[unclaimed_collection] = self.db[unclaimed_collection]

    # Todo: new insert consistency checking code
    def check_consistency(self):
        pass

    def multi_topic_batch_insert(self, document, topic_collection_name):
        """

        Args:
            document:
            topic_collection_name:

        Returns:

        """
        self.topic_insert_buffer[topic_collection_name].append(document)

        if len(self.topic_insert_buffer[topic_collection_name]) == self.batch_insert_size:
            self.collection[topic_collection_name].insert_many(self.topic_insert_buffer[topic_collection_name])

            self.topic_insert_buffer[topic_collection_name] = []

            if self.verbose:
                print("Inserted a batch of: " + str(self.batch_insert_size) + ' new documents.')
