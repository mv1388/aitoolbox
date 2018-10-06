import tweepy
import json

import AIToolbox.DataManipulation.DataAccess.MongoDB as dba


class TwitterAPIConsumerCredentials:
    def __init__(self, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None):
        """

        Args:
            consumer_key:
            consumer_secret:
            access_token:
            access_token_secret:
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

    def __str__(self):
        return "Consumer Key: " + self.consumer_key + \
               "\nConsumer Secret: " + self.consumer_secret + \
               "\nAccess Token: " + self.access_token + \
               "\nAccess Token Secret: " + self.access_token_secret

    def check_if_info_complete(self):
        """

        Returns:

        """
        return self.consumer_key is not None and \
               self.consumer_secret is not None and \
               self.access_token is not None and \
               self.access_token_secret is not None


class TwitterAPIStreamer:
    def __init__(self, twitter_api_credentials):
        """

        Args:
            twitter_api_credentials:
        """
        if not twitter_api_credentials.check_if_info_complete():
            print("Not all API credentials present in the provided credentials object.")

        self.twitter_api_credentials = twitter_api_credentials

        self.auth = tweepy.OAuthHandler(self.twitter_api_credentials.consumer_key,
                                        self.twitter_api_credentials.consumer_secret)
        self.auth.set_access_token(self.twitter_api_credentials.access_token,
                                   self.twitter_api_credentials.access_token_secret)

        self.stream = None

    def start_streaming(self, listener, stream_filter=None,
                        auto_resume=False):
        """

        Args:
            listener:
            stream_filter:
            auto_resume:

        Returns:

        """

        # while True:
        #     try:
        #         self.stream = tweepy.Stream(self.auth, listener)
        #         if stream_filter is not None:
        #             self.stream.filter(track=stream_filter)
        #     except KeyboardInterrupt:
        #         if type(listener) == SaveToFileListener:
        #             listener.output_file.close()
        #             break
        #     except:
        #         print '\t\tProbably network error'
        #         if auto_resume:
        #             print '\t\tResuming streaming'
        #         else:
        #             break

        # Initial version without stream resume after connection break
        # try:
        #     self.stream = tweepy.Stream(self.auth, listener)
        #     if stream_filter is not None:
        #         self.stream.filter(track=stream_filter)
        # except KeyboardInterrupt:
        #     if type(listener) == SaveToFileListener:
        #         listener.output_file.close()

        try:
            self.stream = tweepy.Stream(self.auth, listener)
            if stream_filter is not None:
                self.stream.filter(track=stream_filter)
        except KeyboardInterrupt:
            if type(listener) == SaveToFileListener:
                listener.output_file.close()
        except:
            print('\t\tNetwork connection problem')

            if auto_resume:
                print('\t\tResuming streaming')
                self.start_streaming(listener, stream_filter, auto_resume)

        # if auto_resume:
        #     while True:
        #         try:
        #             self.stream = tweepy.Stream(self.auth, listener)
        #             if stream_filter is not None:
        #                 self.stream.filter(track=stream_filter)
        #         except KeyboardInterrupt:
        #             if type(listener) == SaveToFileListener:
        #                 listener.output_file.close()
        #                 break
        #         except:
        #             print '\t\tProbably network error, resuming streaming'
        # else:
        #     try:
        #         self.stream = tweepy.Stream(self.auth, listener)
        #
        #         if stream_filter is not None:
        #             self.stream.filter(track=stream_filter)
        #     except KeyboardInterrupt:
        #         if type(listener) == SaveToFileListener:
        #             listener.output_file.close()


class PrintOutputListener(tweepy.StreamListener):
    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        text = json.loads(raw_data)

        try:
            print((text["text"]))
        except KeyError:
            print("NO TEXT")

        return True

    def on_error(self, status_code):
        print("ERROR occurred!")
        print(status_code)
        return False

    def on_limit(self, track):
        print("LIMIT occurred!")
        print(track)
        return False

    def on_timeout(self):
        print("\t\t\t\t\tTimeout occurred!")
        return False

    def on_disconnect(self, notice):
        print("DISCONNECT")
        print(notice)
        return False


class SaveToFileListener(PrintOutputListener):
    def __init__(self, output_file_path):
        """

        Args:
            output_file_path:
        """
        super(SaveToFileListener, self).__init__()  # Check this....

        self.file_path = output_file_path

        self.output_file = open(self.file_path, "a")

    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        text = json.loads(raw_data)

        try:
            print((text["text"]))
        except KeyError:
            print("NO TEXT")

        self.output_file.write(raw_data)

        return True


class SaveToMongoDBListener(PrintOutputListener):
    def __init__(self, db_name, collection_name, verbose=False, output_tweets=False):
        """

        Args:
            db_name:
            collection_name:
            verbose:
            output_tweets:
        """
        super(SaveToMongoDBListener, self).__init__()  # Check this....
        self.output_count_bool = verbose
        self.output_count = 0
        self.output_tweets = output_tweets

        self.db_name = db_name
        self.collection_name = collection_name

        self.mongo_db_access = dba.MongoDBWriter(self.db_name, self.collection_name)

    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        json_data = json.loads(raw_data)

        self.mongo_db_access.collection.insert_one(json_data)

        if self.output_tweets:
            try:
                print((json_data['text']))
            except KeyError:
                print("NO TEXT")

        if self.output_count_bool:
            self.output_count += 1

            if self.output_count % 100 == 0:
                print('Number of tweets saved: ' + str(self.output_count))

        return True


class SaveBatchToMongoDBListener(PrintOutputListener):
    def __init__(self, db_name, collection_name, batch_insert_size=10,
                 verbose=False, output_tweets=False):
        """

        Args:
            db_name:
            collection_name:
            batch_insert_size:
            verbose:
            output_tweets:
        """
        super(SaveBatchToMongoDBListener, self).__init__()  # Check this....
        self.verbose = verbose
        self.output_count = 0
        self.output_tweets = output_tweets

        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_insert_size = batch_insert_size

        self.mongo_db_access = dba.MongoDBWriter(self.db_name, self.collection_name, self.batch_insert_size,
                                                 verbose=verbose)

    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        json_data = json.loads(raw_data)

        self.mongo_db_access.batch_insert(json_data)

        if self.output_tweets:
            try:
                print((json_data['text']))
            except KeyError:
                print("NO TEXT")

        if self.verbose:
            self.output_count += 1

            if self.output_count % self.batch_insert_size == 0:
                print('\tNumber of tweets saved: ' + str(self.output_count))

        return True


class SaveMultipleTopicsBatchToMongoDBListener(PrintOutputListener):
    def __init__(self, db_name, collections_topics, unclaimed_collection,
                 batch_insert_size=10, disregard_caps=True,
                 verbose=False, output_tweets=False):
        """

        Args:
            db_name:
            collections_topics:
            unclaimed_collection:
            batch_insert_size:
            disregard_caps:
            verbose:
            output_tweets:
        """
        super(SaveMultipleTopicsBatchToMongoDBListener, self).__init__()  # Check this....
        self.verbose = verbose
        self.output_count = 0
        self.output_tweets = output_tweets

        self.db_name = db_name
        self.collections_topics = collections_topics
        self.unclaimed_collection = unclaimed_collection
        self.batch_insert_size = batch_insert_size
        self.disregard_caps = disregard_caps

        self.mongo_db_multi_topic_access = dba.MongoDBMultiTopicWriter(self.db_name,
                                                                       self.collections_topics,
                                                                       self.unclaimed_collection,
                                                                       self.batch_insert_size, verbose=verbose)

        if self.disregard_caps:
            for topic_collection_name in self.collections_topics:
                self.collections_topics[topic_collection_name] = [topic_el.lower()
                                                                  for topic_el in
                                                                  self.collections_topics[topic_collection_name]]

    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        json_data = json.loads(raw_data)

        found_destination = False

        for topic_collection_name in self.collections_topics:
            current_topic_found = False

            # Check for tweet text
            if 'text' in json_data and \
                    (len([topic_el
                     for topic_el in self.collections_topics[topic_collection_name]
                     if topic_el in
                     (json_data['text'].lower() if
                      self.disregard_caps else json_data['text'])]) > 0):
                current_topic_found = True

            # Check for re-tweets text
            if 'retweeted_status' in json_data and \
                    (len([topic_el
                     for topic_el in self.collections_topics[topic_collection_name]
                     if topic_el in
                         (json_data['retweeted_status']['text'].lower() if
                          self.disregard_caps else json_data['retweeted_status']['text'])]) > 0 or
                        ('quoted_status' in json_data['retweeted_status'] and
                            len([topic_el
                                 for topic_el in self.collections_topics[topic_collection_name]
                                 if topic_el in
                                 (json_data['retweeted_status']['quoted_status']['text'].lower()
                                  if self.disregard_caps else
                                  json_data['retweeted_status']['quoted_status']['text'])]) > 0)
                     ):
                current_topic_found = True

            # Check for quoted status text
            if 'quoted_status' in json_data and \
                len([topic_el
                     for topic_el in self.collections_topics[topic_collection_name]
                     if topic_el in (json_data['quoted_status']['text'].lower()
                                     if self.disregard_caps else
                                     json_data['quoted_status']['text'])]) > 0:
                current_topic_found = True

            if current_topic_found:
                self.mongo_db_multi_topic_access.multi_topic_batch_insert(json_data, topic_collection_name)
                found_destination = True

        if found_destination is False:
            self.mongo_db_multi_topic_access.multi_topic_batch_insert(json_data, self.unclaimed_collection)

        if self.output_tweets:
            try:
                print((json_data['text']))
            except KeyError:
                print("NO TEXT")

        if self.verbose:
            self.output_count += 1

            if self.output_count % self.batch_insert_size == 0:
                print('\tNumber of tweets saved: ' + str(self.output_count))

            if found_destination is False:
                print("Didn't find the topic destination")
                try:
                    print((json_data['text']))
                except KeyError:
                    print("NO TEXT")

        return True


class StreamUserExploreListener(PrintOutputListener):
    def __init__(self, rest_consumer, user_tweets_number,
                 stream_output_file_path, user_posts_output_folder_path,
                 verbose=False):
        """

        Args:
            rest_consumer:
            user_tweets_number:
            stream_output_file_path:
            user_posts_output_folder_path:
            verbose:
        """
        super(StreamUserExploreListener, self).__init__()  # Check this....
        self.verbose = verbose

        self.rest_consumer = rest_consumer
        self.user_tweets_number = user_tweets_number

        self.stream_output_file_path = stream_output_file_path
        self.user_posts_output_folder_path = user_posts_output_folder_path

        self.stream_output = open(self.stream_output_file_path, "a")

    def on_data(self, raw_data):
        """

        Args:
            raw_data:

        Returns:

        """
        self.stream_output.write(raw_data)

        json_text = json.loads(raw_data)
        current_status = tweepy.Status.parse(self.api, json_text)
        current_user_id = current_status.author.id_str

        current_user_tweets = self.rest_consumer.get_user_time_line(current_user_id, self.user_tweets_number)

        current_user_output_file = open(self.user_posts_output_folder_path + current_user_id + ".txt", "a")

        for current_tweet in current_user_tweets:
            current_user_output_file.write(str(current_tweet._json))

        current_user_output_file.close()

        if self.verbose:
            print(current_status.text)
            print(current_user_tweets)
            print("\n")


class TwitterAPIREST:
    def __init__(self, twitter_api_credentials):
        """

        :param twitter_api_credentials:
        """
        if not twitter_api_credentials.check_if_info_complete():
            print("Not all API credentials present in the provided credentials object.")

        self.twitter_api_credentials = twitter_api_credentials

        self.auth = tweepy.OAuthHandler(self.twitter_api_credentials.consumer_key,
                                        self.twitter_api_credentials.consumer_secret)
        self.auth.set_access_token(self.twitter_api_credentials.access_token,
                                   self.twitter_api_credentials.access_token_secret)

        self.api = tweepy.API(self.auth)

    def get_user_time_line(self, user_id, tweet_num_bound):
        """

        :param user_id:
        :param tweet_num_bound:
        :return:
        """
        return self.api.user_timeline(user_id, count=tweet_num_bound)

    def get_user_details(self, user_id):
        pass
