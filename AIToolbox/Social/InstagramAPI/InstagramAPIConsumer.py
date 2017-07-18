import instagram


class InstagramAPIConsumerCredentials:
    def __init__(self, client_id=None, client_secret=None, access_token=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token

    def __str__(self):
        return "Consumer Key: " + self.client_id + \
               "\nConsumer Secret: " + self.client_secret + \
               "\nAccess Token: " + self.access_token

    def check_if_info_complete(self):
        return self.client_id is not None and \
               self.client_secret is not None and \
               self.access_token is not None


class InstagramAPIConsumer:
    def __init__(self):
        pass

