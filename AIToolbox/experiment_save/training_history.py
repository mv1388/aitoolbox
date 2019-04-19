
class TrainingHistory:
    def __init__(self, has_validation=True, auto_epoch='loss', strict_content_check=False):
        """Training history abstraction for storing all the produced model performance evaluations during the training
            process
        
        Args:
            has_validation: if train history should by default include 'val_loss'. This is needed when train loops
                by default evaluate loss on validation set when such a set is available.
            auto_epoch (str): based on which recorded metric in train history, the epoch list is automatically built.
            strict_content_check (bool):
        """
        self.train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if has_validation \
            else {'loss': [], 'accumulated_loss': []}
        
        self.epoch = []

        self.train_history_record = None
        self.auto_epoch = auto_epoch

        self.strict_content_check = strict_content_check
        self.empty_train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if has_validation \
            else {'loss': [], 'accumulated_loss': []}
        
    def insert_single_result_into_history(self, metric_name, metric_result, epoch=None):
        """Insert a new model performance metric evaluation result into the training history

        Args:
            metric_name (str): name of the metric to be stored.
            metric_result (float or dict): metric performance result to be stored.
            epoch (int): manually specified epoch idx. Important to note is that out of all the metrics that are
                recorded in every epoch, only one metric insertion per epoch should manually set epoch parameter.
                This however disables automatic epoch deduction based on auto_epoch class parameter. User should thus
                be careful when using epoch parameter and should rather use the auto_epoch option in most cases to
                ensure the expected behaviour.
        """
        if metric_name not in self.train_history:
            self.train_history[metric_name] = []
        self.train_history[metric_name].append(metric_result)
        
        if epoch is not None and (len(self.epoch) == 0 or epoch > max(self.epoch)):
            self.epoch.append(epoch)

    def _build_epoch_list(self):
        """

        Returns:
            list:
        """
        return list(range(len(self.train_history[self.auto_epoch]))) if len(self.epoch) == 0 \
            else self.epoch
        
    def get_train_history(self):
        """

        Returns:
            dict:
        """
        self.epoch = self._build_epoch_list()
        self.train_history_record = {'history': self.train_history, 'epoch': self.epoch}

        return self.train_history_record

    def get_train_history_dict(self):
        """

        Returns:
            dict:
        """
        if self.train_history == self.empty_train_history:
            self.warn_about_result_data_problem('Train History dict is empty')

        return self.train_history

    def get_epoch_list(self):
        """

        Returns:
            list:
        """
        return self._build_epoch_list()

    def wrap_pre_prepared_history(self, history, epoch):
        """

        Args:
            history (dict): 
            epoch (list): 

        Returns:
            self
            
        Examples:
            train_history = model.fit(x_train, y_train, ... )
            history = train_history.history
            epoch = train_history.epoch

            # history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298, 1.039527416229248,
                0.9796673059463501], '
                    val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                0.5799999833106995,  0.46000000834465027, 0.699999988079071, 0.7599999904632568, 0.7200000286102295],
                    'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867, 0.8969314098358154],
                    'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568, 0.6700000166893005,
                0.7599999904632568]}
            # epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        self.train_history = history
        self.epoch = epoch
        return self

    def qa_check_history_records(self):
        """

        Returns:
            None
        """
        for k in self.train_history:
            if len(self.train_history[k]) != len(self.epoch):
                self.warn_about_result_data_problem(
                    f'Warning: Train history records not of the same size. Problem with: {k}')

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

    def __str__(self):
        return f'{self.train_history}'

    def __len__(self):
        return len(self.train_history)
    
    def __getitem__(self, item):
        return self.train_history[item]

    def __setitem__(self, key, value):
        self.insert_single_result_into_history(key, value)

    def __contains__(self, item):
        return item in self.train_history

    def __iter__(self):
        for k in self.train_history:
            yield k

    def keys(self):
        return self.train_history.keys()

    def items(self):
        return self.train_history.items()
