import AIToolbox.Social.TwitterAPI.TwitterAPIConsumer as tac

consumer_key = "Uxke6pw6Q56l16olV5iiT4q4F"
consumer_secret = "AxHr3elTSe4AeDyb1apBtJ7qlZbmc0NghpYx76djtyl9UoP74D"
access_token = "3091794297-OF0GaRr6GFe8wvICKqJZXKbUmiUKvA5uutf2uSZ"
access_token_secret = "2Ne2LaLo7ZK5tyNnbVzLOgzO3dYAfqbfNlTVV6PDvBF4Z"

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)

streamer = tac.TwitterAPIStreamer(creds)
rest_consumer = tac.TwitterAPIREST(creds)

user_explore_listener = tac.StreamUserExploreListener(rest_consumer, 50, "data/stream_output.txt", "data/",
                                                      verbose=False)

try:
    streamer.start_streaming(user_explore_listener, ["rain"])
except KeyboardInterrupt:
    user_explore_listener.stream_output.close()
