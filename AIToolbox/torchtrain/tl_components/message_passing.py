
class MessageService:
    def __init__(self, auto_purge_on_update=True):
        self.auto_purge_on_update = auto_purge_on_update

        self.message_store = {}

        self.message_handling_settings = {}

    def end_of_epoch_trigger(self):
        pass

    def read_messages(self, key):
        if key in self.message_store and len(self.message_store[key]) > 0:
            return self.message_store[key] if len(self.message_store[key]) > 1 else self.message_store[key][0]
        else:
            return None

    def write_message(self, key, value, msg_handling_setting=None):
        self.auto_purge_messages(key, msg_handling_setting)

        if key not in self.message_store:
            self.message_store[key] = []

        self.message_store[key].append(value)

    def auto_purge_messages(self, key, purge_manual_setting):
        if purge_manual_setting or (purge_manual_setting is None and self.auto_purge_on_update):
            self.purge_messages_by_key(key)

    def purge_messages_by_key(self, key):
        if key in self.message_store:
            del self.message_store[key]
