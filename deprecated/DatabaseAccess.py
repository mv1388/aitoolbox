from pymongo import MongoClient


"""
    OLD code
    
    still used database API for social.twitter streamer
    
    Don't delete until twitter streamer is rewritten for data_access library

"""


class MongoDBWriter:
    def __init__(self, db_name, collection_name, batch_insert_size=1, verbose=False):
        """

        Args:
            db_name:
            collection_name:
            batch_insert_size:
            verbose:
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
            db_name:
            collections_topics:
            unclaimed_collection:
            batch_insert_size:
            verbose:
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


class DBConnection:
    def __init__(self, user, password):
        pass


class MySQLAccess:
    def __init__(self):
        pass


class SQLLiteAccess:
    def __init__(self):
        pass
