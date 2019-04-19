

class TrainingHistory:
    def __init__(self, history, epoch, strict_content_check=False):
        """

        In pytorch there is no pre-prepared history and epoch functionality. You will probably have to construct the
        history dict by yourself.

        Args:
            history (dict):
            epoch (list):
            strict_content_check (bool):

        Examples:
            train_history = model.fit(x_train, y_train, ... )
            history = train_history.history
            epoch = train_history.epoch

            # history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494, 1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298, 1.039527416229248, 0.9796673059463501], 'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721, 0.5799999833106995, 0.46000000834465027, 0.699999988079071, 0.7599999904632568, 0.7200000286102295], 'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588, 1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867, 0.8969314098358154], 'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705, 0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568, 0.6700000166893005, 0.7599999904632568]}
            # epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        self.history = history
        self.epoch = epoch
        self.strict_content_check = strict_content_check

        self.train_history_dict = None
        self.prepare_train_history_dict()

    def prepare_train_history_dict(self):
        """

        Returns:
            None
        """
        self.train_history_dict = {'history': self.history,
                                   'epoch': self.epoch}

    def get_train_history(self):
        """
        
        Returns:
            dict:
        """
        if self.train_history_dict is None:
            self.warn_about_result_data_problem('Warning: Train history dict missing')
        return self.train_history_dict
    
    def qa_check_history_records(self):
        """
        
        Returns:
            None
        """
        for k in self.history:
            if len(self.history[k]) != len(self.epoch):
                self.warn_about_result_data_problem(f'Warning: Train history records not of the same size. Problem with: {k}')

    def warn_about_result_data_problem(self, msg):
        """

        Args:
            msg (str):

        Returns:
            None
        """
        if self.strict_content_check:
            raise ValueError(msg)
        else:
            print(msg)
