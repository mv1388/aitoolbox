import AIToolbox.social.TwitterAPIConsumer as tac

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)

streamer = tac.TwitterAPIStreamer(creds)
rest_consumer = tac.TwitterAPIREST(creds)

user_explore_listener = tac.StreamUserExploreListener(rest_consumer, 50, "data/stream_output.txt", "data/",
                                                      verbose=False)

try:
    streamer.start_streaming(user_explore_listener, ["rain"])
except KeyboardInterrupt:
    user_explore_listener.stream_output.close()
