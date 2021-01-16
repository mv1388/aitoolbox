import unittest

from aitoolbox.torchtrain.train_loop.components import message_passing as msg_pass


class TestMessageService(unittest.TestCase):
    def test_message_handling_labels(self):
        self.assertEqual(msg_pass.KEEP_FOREVER, 'keep_forever')
        self.assertEqual(msg_pass.UNTIL_END_OF_EPOCH, 'until_end_of_epoch')
        self.assertEqual(msg_pass.UNTIL_READ, 'until_read')

    def test_init(self):
        msg_service = msg_pass.MessageService()
        self.assertEqual(msg_service.message_store, {})

    def test_read_messages(self):
        msg_service = msg_pass.MessageService()
        msg_service.write_message('test_float', 344.452)
        msg_service.write_message('test_float', 1000, msg_pass.UNTIL_READ)
        msg_service.write_message('test_float', 123)
        msg_service.write_message('test_list_float', [433.44, 111.3322, 0.4])
        msg_service.write_message('test_list_float', [1919191, 333], msg_pass.UNTIL_READ)
        msg_service.write_message('test_list_float', [494.53, 2.], msg_pass.UNTIL_READ)
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_READ]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4],
                                               [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        read_msgs = msg_service.read_messages('test_float')
        self.assertEqual(read_msgs, [344.452, 1000, 123])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4],
                                               [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        read_msgs = msg_service.read_messages('test_float')
        self.assertEqual(read_msgs, [344.452, 123])
        read_msgs_2 = msg_service.read_messages('test_list_float')
        self.assertEqual(read_msgs_2, [[433.44, 111.3322, 0.4], [1919191, 333], [494.53, 2.]])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              (
                                                  'test_list_float', [433.44, 111.3322, 0.4],
                                                  [msg_pass.UNTIL_END_OF_EPOCH])
                                          ]})

        self.assertIsNone(msg_service.read_messages('non_existing_msg'))

    def test_write_message(self):
        msg_service = msg_pass.MessageService()

        msg_service.write_message('test_float', 344.452)
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH])]})

        msg_service.write_message('test_float', 1000)
        msg_service.write_message('test_float', 123)
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])]})

        msg_service.write_message('test_list_float', [433.44, 111.3322, 0.4])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              (
                                                  'test_list_float', [433.44, 111.3322, 0.4],
                                                  [msg_pass.UNTIL_END_OF_EPOCH])
                                          ]})

        msg_service.write_message('test_list_float', [1919191, 333])
        msg_service.write_message('test_list_float', [494.53, 2.])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4],
                                               [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_END_OF_EPOCH])]})

        msg_service.write_message('test_overwrite', [1919191, 333], [msg_pass.OVERWRITE])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4],
                                               [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_overwrite': [
                                              ('test_overwrite', [1919191, 333], [msg_pass.OVERWRITE])
                                          ]})

        msg_service.write_message('test_overwrite', [1, 2], [msg_pass.OVERWRITE])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4],
                                               [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_END_OF_EPOCH]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_overwrite': [
                                              ('test_overwrite', [1, 2], [msg_pass.OVERWRITE])
                                          ]})

        with self.assertRaises(ValueError):
            msg_service.write_message('test_list_float', [1919191, 333], 'unsupported_setting')

    def test_write_message_handling_settings(self):
        msg_service = msg_pass.MessageService()

        msg_service.write_message('test_float', 344.452, [msg_pass.UNTIL_READ, msg_pass.OVERWRITE])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [
                                             ('test_float', 344.452, [msg_pass.UNTIL_READ, msg_pass.OVERWRITE])]})

        msg_service.write_message('test_float', 22222222, [msg_pass.UNTIL_READ, msg_pass.OVERWRITE])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [
                                             ('test_float', 22222222, [msg_pass.UNTIL_READ, msg_pass.OVERWRITE])]})

        msg_service.read_messages('test_float')
        self.compare_msg_service_content(msg_service, {'test_float': []})

        msg_service.write_message('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])
        msg_service.write_message('test_float', 456, [msg_pass.UNTIL_END_OF_EPOCH, msg_pass.OVERWRITE])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [
                                             ('test_float', 456, [msg_pass.UNTIL_END_OF_EPOCH, msg_pass.OVERWRITE])]})

        msg_service.end_of_epoch_trigger()
        self.compare_msg_service_content(msg_service, {})

        with self.assertRaises(ValueError):
            msg_service.write_message('test_list_float', [1919191, 333],
                                      [msg_pass.UNTIL_END_OF_EPOCH, msg_pass.UNTIL_READ])

    def test_end_of_epoch_trigger(self):
        msg_service = msg_pass.MessageService()
        msg_service.write_message('test_float', 344.452)
        msg_service.write_message('test_float', 1000, [msg_pass.UNTIL_READ])
        msg_service.write_message('test_float', 123)
        msg_service.write_message('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER])
        msg_service.write_message('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ])
        msg_service.write_message('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 344.452, [msg_pass.UNTIL_END_OF_EPOCH]),
                                                         ('test_float', 1000, [msg_pass.UNTIL_READ]),
                                                         ('test_float', 123, [msg_pass.UNTIL_END_OF_EPOCH])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        msg_service.end_of_epoch_trigger()
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [('test_float', 1000, [msg_pass.UNTIL_READ])],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        msg_service.read_messages('test_float')
        self.compare_msg_service_content(msg_service,
                                         {'test_float': [],
                                          'test_list_float': [
                                              ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER]),
                                              ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                              ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        msg_service.end_of_epoch_trigger()
        self.compare_msg_service_content(msg_service,
                                         {'test_list_float': [
                                             ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER]),
                                             ('test_list_float', [1919191, 333], [msg_pass.UNTIL_READ]),
                                             ('test_list_float', [494.53, 2.], [msg_pass.UNTIL_READ])]})

        msg_service.read_messages('test_list_float')
        msg_service.write_message('test_list_float', [343, 34])
        msg_service.write_message('completely_new', [1111, 34222])
        self.compare_msg_service_content(msg_service,
                                         {'completely_new': [
                                             ('completely_new', [1111, 34222], [msg_pass.UNTIL_END_OF_EPOCH])],
                                             'test_list_float': [
                                                 ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER]),
                                                 ('test_list_float', [343, 34], [msg_pass.UNTIL_END_OF_EPOCH])]})

        msg_service.end_of_epoch_trigger()
        self.compare_msg_service_content(msg_service,
                                         {'test_list_float': [
                                             ('test_list_float', [433.44, 111.3322, 0.4], [msg_pass.KEEP_FOREVER])]})

    def compare_msg_service_content(self, msg_service, expected_msg_service):
        msg_service_content = {}

        for k, v_list in msg_service.message_store.items():
            msg_service_content[k] = [(msg.key, msg.value, msg.msg_handling_settings) for msg in v_list]

        self.assertEqual(sorted(list(msg_service.message_store.keys())),
                         sorted(list(expected_msg_service.keys())))

        for k in expected_msg_service:
            self.assertEqual(msg_service_content[k], expected_msg_service[k])
