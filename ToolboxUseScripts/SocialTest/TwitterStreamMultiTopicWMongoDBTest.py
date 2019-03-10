import AIToolbox.social.TwitterAPIConsumer as tac


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)

streamer = tac.TwitterAPIStreamer(creds)

collections_topics = {'test_multi_elections': ['Donald', 'Trump', 'Donald Trump', 'donaldtrump', 'donald_trump',
                                             'Hillary', 'Clinton', 'Hillary Clinton', 'hillaryclinton',
                                             'hillary_clinton',
                                             'Barack', 'Obama', 'Barack Obama', 'barackobama', 'barack_obama',
                                             'GOP', 'DNC'],
                      'test_multi_GOT': ['game of thrones', 'gameofthrones']}

mongo_multi_topic_listener = tac.SaveMultipleTopicsBatchToMongoDBListener('test',
                                                                          collections_topics,
                                                                          'test_u_c',
                                                                          batch_insert_size=100,
                                                                          verbose=True, output_tweets=False)

streamer.start_streaming(mongo_multi_topic_listener, ['Donald', 'Trump', 'Donald Trump', 'donaldtrump', 'donald_trump',
                                                      'Hillary', 'Clinton', 'Hillary Clinton', 'hillaryclinton',
                                                      'hillary_clinton',
                                                      'Barack', 'Obama', 'Barack Obama', 'barackobama', 'barack_obama',
                                                      'GOP', 'DNC',
                                                      'game of thrones', 'gameofthrones'])
