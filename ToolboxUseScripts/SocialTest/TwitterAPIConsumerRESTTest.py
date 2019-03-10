import AIToolbox.social.TwitterAPIConsumer as tac


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

creds = tac.TwitterAPIConsumerCredentials(consumer_key, consumer_secret, access_token, access_token_secret)

rest_consumer = tac.TwitterAPIREST(creds)

public_tweets = rest_consumer.api.user_timeline("1591684358", count=50)
for tweet in public_tweets:
    print(tweet)


# trends = api.trends_available()
# trends = api.trends_place(1)
