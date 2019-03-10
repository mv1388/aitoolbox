import AIToolbox.social.TwitterAPIConsumer as tac


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)
print_listener = tac.PrintOutputListener()

streamer = tac.TwitterAPIStreamer(creds)
streamer.start_streaming(print_listener, ["top gear"])
