import pickle


class BasicResultSaver:
    def __init__(self, result_dump_path):
        """

        Args:
            result_dump_path (str):
        """
        self.result_dump_path = result_dump_path

    def record_result(self, experiment_descr, training_history, test_loss, test_acc):
        data_record = {'experiment_description': experiment_descr,
                       'train_hostory': training_history,
                       'test_loss_acc': (test_loss, test_acc)}

        pickle.dump(data_record, open(self.result_dump_path, 'wb'))
