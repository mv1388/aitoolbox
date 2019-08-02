KEEP_FOREVER = 'keep_forever'
UNTIL_END_OF_EPOCH = 'until_end_of_epoch'
UNTIL_READ = 'until_read'


class MessageService:
    def __init__(self):
        self.message_store = {}
        self.message_handling_settings = {}

    def read_messages(self, key):
        """

        Args:
            key (str):

        Returns:
            list or None:
        """
        if key in self.message_store:
            messages = self.message_store[key]

            if self.message_handling_settings[key] == UNTIL_READ:
                del self.message_store[key]

            return messages
        else:
            return None

    def write_message(self, key, value, msg_handling_setting=UNTIL_END_OF_EPOCH):
        """

        Args:
            key (str):
            value:
            msg_handling_setting (str):

        Returns:
            None
        """
        if key not in self.message_handling_settings:
            self.message_handling_settings[key] = msg_handling_setting

        if key not in self.message_store:
            self.message_store[key] = []

        self.message_store[key].append(value)

    def end_of_epoch_trigger(self):
        for key, msg_handling_rule in self.message_handling_settings.items():
            if key in self.message_store and msg_handling_rule == UNTIL_END_OF_EPOCH:
                del self.message_store[key]
