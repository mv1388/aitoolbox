from enum import Enum


class MessageHandling(Enum):
    KEEP_FOREVER = 'keep_forever'
    UNTIL_END_OF_EPOCH = 'until_end_of_epoch'
    UNTIL_READ = 'until_read'
    OVERWRITE = 'overwrite'


class Message:
    def __init__(self, key, value, msg_handling_settings):
        """Wrapper object to represent the messages in the MessageService together with their handling settings

        Args:
            key (str): message key
            value: message value
            msg_handling_settings (MessageHandling or list[MessageHandling]): selected message handling settings
                for this particular message
        """
        self.key = key
        self.value = value
        self.msg_handling_settings = msg_handling_settings \
            if type(msg_handling_settings) is list else [msg_handling_settings]


class MessageService:
    def __init__(self):
        """Message Passing Service

        Primarily intended for passing the messages in the TrainLoop, especially for communication or data sharing
        between different callbacks.
        """
        self.message_store = {}

    def read_messages(self, key):
        """Read messages by key from the TrainLoop message service

        Args:
            key (str): message key

        Returns:
            list or None: if message key present return content, otherwise return None
        """
        if key in self.message_store:
            messages = [msg.value for msg in self.message_store[key]]
            self.message_store[key] = [
                msg for msg in self.message_store[key]
                if MessageHandling.UNTIL_READ not in msg.msg_handling_settings
            ]
            return messages
        else:
            return None

    def write_message(self, key, value, msg_handling_settings=MessageHandling.UNTIL_END_OF_EPOCH):
        """Write a new message to the message service

        Args:
            key (str): message key
            value: message content
            msg_handling_settings (MessageHandling or list[MessageHandling]): setting how to handle the lifespan of
                the message. Can use one of the following message lifecycle handling settings which are variables
                imported from this script file and can be found defined at the beginning of the script:

                * ``KEEP_FOREVER``
                * ``UNTIL_END_OF_EPOCH``
                * ``UNTIL_READ``
                * ``OVERWRITE``

        Returns:
            None
        """
        msg_handling_settings = self.validate_msg_handling_settings(msg_handling_settings)

        if key not in self.message_store:
            self.message_store[key] = []

        message = Message(key, value, msg_handling_settings)

        if MessageHandling.OVERWRITE in msg_handling_settings:
            self.message_store[key] = [message]
        else:
            self.message_store[key].append(message)

    def end_of_epoch_trigger(self):
        """Purging of the message service at the end of the epoch

        Normally executed by the TrainLoop automatically after all the callbacks were executed at the end of every epoch

        Returns:
            None
        """
        for key, msgs_list in list(self.message_store.items()):
            self.message_store[key] = [msg for msg in self.message_store[key]
                                       if MessageHandling.UNTIL_END_OF_EPOCH not in msg.msg_handling_settings]

            if len(self.message_store[key]) == 0:
                del self.message_store[key]

    @staticmethod
    def validate_msg_handling_settings(msg_handling_settings):
        if type(msg_handling_settings) == list:
            for msg_setting in msg_handling_settings:
                if type(msg_setting) != MessageHandling:
                    raise TypeError('msg_setting is not of the correct MessageHandling type. '
                                    f'It is {type(msg_setting)}.')

            if len(msg_handling_settings) > 1 and MessageHandling.OVERWRITE not in msg_handling_settings:
                raise ValueError(f'Provided two incompatible msg_handling_settings {msg_handling_settings}. '
                                 'Only OVERRIDE setting can currently be combined with another available setting.')
        elif type(msg_handling_settings) != MessageHandling:
            raise TypeError(f'Provided msg_handling_settings {msg_handling_settings} type not of the supported '
                            'MessageHandling or list of MessageHandling.')

        return msg_handling_settings if type(msg_handling_settings) is list else [msg_handling_settings]
