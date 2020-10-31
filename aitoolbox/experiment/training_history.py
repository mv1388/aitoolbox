import copy
from aitoolbox.utils import dict_util


class TrainingHistory:
    def __init__(self, has_validation=True, strict_content_check=False):
        """Training history abstraction adding specific functionality to the simple dict

        In many ways the object can be used with the same API as a normal python dict. However, for the need of
        tracking performance in the TrainLoop TrainingHistory offers additional functions handling the input, output
        and quality assurance of the stored results.

        Args:
            has_validation: if train history should by default include 'val_loss'. This is needed when train loops
                by default evaluate loss on validation set when such a set is available.
            strict_content_check (bool): should just print warning or raise the error and crash in case of found
                (quality) problems
        """
        self.train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if has_validation \
            else {'loss': [], 'accumulated_loss': []}

        self.strict_content_check = strict_content_check
        self.empty_train_history = {'loss': [], 'accumulated_loss': [], 'val_loss': []} if has_validation \
            else {'loss': [], 'accumulated_loss': []}
        
    def insert_single_result_into_history(self, metric_name, metric_result):
        """Insert a key-value formatted result into the training history

        Args:
            metric_name (str): name of the metric to be stored.
            metric_result (float or dict): metric performance result to be stored.
        """
        if metric_name not in self.train_history:
            self.train_history[metric_name] = []
        self.train_history[metric_name].append(metric_result)
        
    def get_train_history(self):
        """Returns the whole train history dict in its original form without any transformations

        Returns:
            dict: training history dict
        """
        return self.train_history

    def get_train_history_dict(self, flatten_dict=False):
        """Returns QA-ed and optionally flattened training history dict

        Args:
            flatten_dict (bool): should the returned training history dict be flattened. So no nested dicts of dicts.
                The keys of the nested dicts will we "_" concatenated and moved into the single level dict.

        Returns:
            dict: training history dict
        """
        if self.train_history == self.empty_train_history:
            self.warn_about_result_data_problem('Train History dict is empty')

        return dict_util.flatten_combine_dict(self.train_history) if flatten_dict else self.train_history

    def wrap_pre_prepared_history(self, history):
        """Wrap existing history dict into the TrainingHistory object

        Args:
            history (dict): training history base dict

        Returns:
            TrainingHistory: self
            
        Examples:
            Expected history dict to be wrapped:

            .. code-block:: python

                history = {
                    'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                 1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                 1.039527416229248, 0.9796673059463501],
                    'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                                0.5799999833106995,  0.46000000834465027, 0.699999988079071, 0.7599999904632568,
                                0.7200000286102295],
                    'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                             1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                             0.8969314098358154],
                    'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                            0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                            0.6700000166893005, 0.7599999904632568]
                }
        """
        self.train_history = history
        return self

    def qa_check_history_records(self):
        """Quality check history

        Returns:
            None
        """
        accepted_len = self.train_history[list(self.train_history.keys())[0]]

        for k in self.train_history:
            if len(self.train_history[k]) != accepted_len:
                self.warn_about_result_data_problem(
                    f'Warning: Train history records not of the same size. Problem with: {k}')

    def warn_about_result_data_problem(self, msg):
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

    def __add__(self, other):
        self_copy = copy.deepcopy(self)
        self_copy.add_history_dict(other)
        return self_copy

    def __radd__(self, other):
        self_copy = copy.deepcopy(self)
        self_copy.add_history_dict(other)
        return self_copy

    def __iadd__(self, other):
        self.add_history_dict(other)
        return self

    def add_history_dict(self, other):
        """Add another training history dict to this training history

        Args:
            other (dict): another training history dict

        Returns:
            None
        """
        if type(other) is not dict:
            raise TypeError(f'Other should be dict. Provided: {type(other)}')

        for k, v in other.items():
            self.insert_single_result_into_history(k, v)
