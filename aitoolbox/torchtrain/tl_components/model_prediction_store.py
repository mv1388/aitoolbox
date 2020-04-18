
class ModelPredictionStore:
    def __init__(self, auto_purge=False):
        """Service for TrainLoop enabling the prediction caching

        Prediction calculation can be costly and it can have severe performance implications if the same predictions
        would be calculated repeatedly. This store caches already made predictions in the current epoch of
        the TrainLoop which takes the cached values if they are available instead of recalculating.

        Args:
            auto_purge (bool): should the prediction service cache be automatically purged at the end of each epoch
        """
        self.do_auto_purge = auto_purge

        self.prediction_store = {'epoch': 0}

    def insert_train_predictions(self, predictions, epoch, force_prediction=False):
        """

        Args:
            predictions (tuple):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('train_pred', predictions, epoch, force_prediction)

    def insert_val_predictions(self, predictions, epoch, force_prediction=False):
        """

        Args:
            predictions (tuple):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('val_pred', predictions, epoch, force_prediction)

    def insert_test_predictions(self, predictions, epoch, force_prediction=False):
        """

        Args:
            predictions (tuple):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('test_pred', predictions, epoch, force_prediction)

    def get_train_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        return self._get_data('train_pred', epoch)

    def get_val_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        return self._get_data('val_pred', epoch)

    def get_test_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            tuple:
        """
        return self._get_data('test_pred', epoch)

    def has_train_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('train_pred', epoch)

    def has_val_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('val_pred', epoch)

    def has_test_predictions(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('test_pred', epoch)

    def insert_train_loss(self, loss, epoch, force_prediction=False):
        """

        Args:
            loss (float):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('train_loss', loss, epoch, force_prediction)

    def insert_val_loss(self, loss, epoch, force_prediction=False):
        """

        Args:
            loss (float):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('val_loss', loss, epoch, force_prediction)

    def insert_test_loss(self, loss, epoch, force_prediction=False):
        """

        Args:
            loss (float):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self._insert_data('test_loss', loss, epoch, force_prediction)

    def get_train_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            float:
        """
        return self._get_data('train_loss', epoch)

    def get_val_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            float:
        """
        return self._get_data('val_loss', epoch)

    def get_test_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            float:
        """
        return self._get_data('test_loss', epoch)

    def has_train_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('train_loss', epoch)

    def has_val_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('val_loss', epoch)

    def has_test_loss(self, epoch):
        """

        Args:
            epoch (int):

        Returns:
            bool:
        """
        return self._has_data('test_loss', epoch)

    def _insert_data(self, source_name, data, epoch, force_prediction=False):
        """

        Args:
            source_name (str):
            data (tuple or float):
            epoch (int):
            force_prediction (bool):

        Returns:
            None
        """
        self.auto_purge(epoch)

        if not self._has_data(source_name, epoch) or force_prediction:
            self.prediction_store[source_name] = data
        else:
            raise ValueError

    def _get_data(self, source_name, epoch):
        """

        Args:
            source_name (str):
            epoch (int):

        Returns:
            tuple or float:
        """
        if self._has_data(source_name, epoch):
            print(f'Getting {source_name} predictions/loss from store')
            return self.prediction_store[source_name]
        else:
            raise ValueError

    def _has_data(self, source_name, epoch):
        """

        Args:
            source_name (str):
            epoch (int):

        Returns:
            bool:
        """
        return source_name in self.prediction_store and epoch == self.prediction_store['epoch']

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
