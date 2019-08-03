KEEP_FOREVER = 'keep_forever'
UNTIL_END_OF_EPOCH = 'until_end_of_epoch'
UNTIL_READ = 'until_read'

ACCEPTED_SETTINGS = [KEEP_FOREVER, UNTIL_END_OF_EPOCH, UNTIL_READ]


class Message:
    def __init__(self, key, value, msg_handling_setting):
        self.key = key
        self.value = value
        self.msg_handling_setting = msg_handling_setting


class MessageService:
    def __init__(self):
        self.message_store = {}

    def read_messages(self, key):
        """

        Args:
            key (str):

        Returns:
            list or None:
        """
        if key in self.message_store:
            messages = [msg.value for msg in self.message_store[key]]
            self.message_store[key] = [msg for msg in self.message_store[key] if msg.msg_handling_setting != UNTIL_READ]
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
        if msg_handling_setting not in ACCEPTED_SETTINGS:
            raise ValueError(f'Provided msg_handling_setting {msg_handling_setting} is not supported. '
                             f'Currently supported settings are: {ACCEPTED_SETTINGS}.')

        if key not in self.message_store:
            self.message_store[key] = []

        message = Message(key, value, msg_handling_setting)
        self.message_store[key].append(message)

    def end_of_epoch_trigger(self):
        """Purging of the message service at the end of the epoch

        Normally executed by the TrainLoop automatically after all the callbacks were executed at the end of every epoch

        Returns:
            None
        """
        for key, msgs_list in list(self.message_store.items()):
            self.message_store[key] = [msg for msg in self.message_store[key]
                                       if msg.msg_handling_setting != UNTIL_END_OF_EPOCH]

            if len(self.message_store[key]) == 0:
                del self.message_store[key]
