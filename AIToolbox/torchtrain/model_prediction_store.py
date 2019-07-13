
class ModelPredictionStore:
    def __init__(self, auto_purge=False):
        """

        Args:
            auto_purge (bool):
        """
        self.do_auto_purge = auto_purge

        self.prediction_store = {'epoch': 0}

    def insert_train_predictions(self, predictions, epoch):
        """

        Args:
            predictions (tuple):
            epoch (int):

        Returns:
            None
        """
        self.auto_purge(epoch)

        if not self.has_train_predictions(epoch):
            self.prediction_store['train_pred'] = predictions
        else:
            raise ValueError

    def insert_val_predictions(self, predictions, epoch):
        """

        Args:
            predictions (tuple):
            epoch (int):

        Returns:
            None
        """
        self.auto_purge(epoch)

        if not self.has_val_predictions(epoch):
            self.prediction_store['val_pred'] = predictions
        else:
            raise ValueError

    def insert_test_predictions(self, predictions, epoch):
        """

        Args:
            predictions (tuple):
            epoch (int):

        Returns:
            None
        """
        self.auto_purge(epoch)

        if not self.has_test_predictions(epoch):
            self.prediction_store['test_pred'] = predictions
        else:
            raise ValueError

    def get_train_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        if epoch == self.prediction_store['epoch'] and self.has_train_predictions(epoch):
            print('Getting train set predictions from store')
            return self.prediction_store['train_pred']
        else:
            raise ValueError

    def get_val_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        if epoch == self.prediction_store['epoch'] and self.has_val_predictions(epoch):
            print('Getting validation set predictions from store')
            return self.prediction_store['val_pred']
        else:
            raise ValueError

    def get_test_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        if epoch == self.prediction_store['epoch'] and self.has_test_predictions(epoch):
            print('Getting test set predictions from store')
            return self.prediction_store['test_pred']
        else:
            raise ValueError

    def has_train_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return 'train_pred' in self.prediction_store and epoch == self.prediction_store['epoch']

    def has_val_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return 'val_pred' in self.prediction_store and epoch == self.prediction_store['epoch']

    def has_test_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return 'test_pred' in self.prediction_store and epoch == self.prediction_store['epoch']

    def auto_purge(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            None
        """
        if self.do_auto_purge and epoch > self.prediction_store['epoch']:
            print('Auto purging prediction store')
            self.purge_prediction_store()

    def purge_prediction_store(self):
        self.prediction_store = {'epoch': self.prediction_store['epoch'] + 1}
