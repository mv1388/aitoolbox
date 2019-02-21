import AIToolbox.social.TwitterAPIConsumer as tac


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)

streamer = tac.TwitterAPIStreamer(creds)

# mongo_listener = tac.SaveToMongoDBListener('test', 'test_twitter_stream', output_tweets=True)
# streamer.start_streaming(mongo_listener, ['Donald', 'Trump', 'Donald Trump', 'donaldtrump'])

mongo_listener = tac.SaveBatchToMongoDBListener('test', 'test_batchtwitter_stream', batch_insert_size=50,
                                                verbose=True, output_tweets=True)
streamer.start_streaming(mongo_listener, ['Donald', 'Trump', 'Donald Trump', 'donaldtrump'])
