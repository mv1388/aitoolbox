from abc import ABCMeta, abstractmethod


class EvalReport(metaclass=ABCMeta):
    @abstractmethod
    def eval_compute(self, *args):
        pass

    @abstractmethod
    def get_eval_record(self):
        pass


class NLPQAEvalReport(EvalReport):
    def __init__(self):
        pass

    def eval_compute(self, *args):
        pass

    def get_eval_record(self):
        """

        Returns:
            dict:

        """
        return {}
