
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
        """Insert training dataset predictions into the cache

        Args:
            predictions (tuple): model training dataset predictions
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('train_pred', predictions, epoch, force_prediction)

    def insert_val_predictions(self, predictions, epoch, force_prediction=False):
        """Insert validation dataset predictions into the cache

        Args:
            predictions (tuple): model validation dataset predictions
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('val_pred', predictions, epoch, force_prediction)

    def insert_test_predictions(self, predictions, epoch, force_prediction=False):
        """Insert test dataset predictions into the cache

        Args:
            predictions (tuple): model test dataset predictions
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('test_pred', predictions, epoch, force_prediction)

    def get_train_predictions(self, epoch):
        """Get training dataset predictions out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            tuple: cached model train dataset predictions
        """
        return self._get_data('train_pred', epoch)

    def get_val_predictions(self, epoch):
        """Get validation dataset predictions out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            tuple: cached model validation dataset predictions
        """
        return self._get_data('val_pred', epoch)

    def get_test_predictions(self, epoch):
        """Get test dataset predictions out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            tuple: cached model test dataset predictions
        """
        return self._get_data('test_pred', epoch)

    def has_train_predictions(self, epoch):
        """Are there training dataset predictions in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('train_pred', epoch)

    def has_val_predictions(self, epoch):
        """Are there validation dataset predictions in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('val_pred', epoch)

    def has_test_predictions(self, epoch):
        """Are there test dataset predictions in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('test_pred', epoch)

    def insert_train_loss(self, loss, epoch, force_prediction=False):
        """Insert training dataset loss into the cache

        Args:
            loss (float):  model train dataset loss
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('train_loss', loss, epoch, force_prediction)

    def insert_val_loss(self, loss, epoch, force_prediction=False):
        """Insert validation dataset loss into the cache

        Args:
            loss (float): model validation dataset loss
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('val_loss', loss, epoch, force_prediction)

    def insert_test_loss(self, loss, epoch, force_prediction=False):
        """Insert test dataset loss into the cache

        Args:
            loss (float): model test dataset loss
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('test_loss', loss, epoch, force_prediction)

    def get_train_loss(self, epoch):
        """Get training dataset model loss out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            float: cached model train dataset loss
        """
        return self._get_data('train_loss', epoch)

    def get_val_loss(self, epoch):
        """Get validation dataset model loss out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            float: cached model validation dataset loss
        """
        return self._get_data('val_loss', epoch)

    def get_test_loss(self, epoch):
        """Get test dataset model loss out of the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            float: cached model test dataset loss
        """
        return self._get_data('test_loss', epoch)

    def has_train_loss(self, epoch):
        """Is there training dataset model loss in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if loss value is in the cache
        """
        return self._has_data('train_loss', epoch)

    def has_val_loss(self, epoch):
        """Is there validation dataset model loss in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if loss value is in the cache
        """
        return self._has_data('val_loss', epoch)

    def has_test_loss(self, epoch):
        """Is there test dataset model loss in the cache

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if loss value is in the cache
        """
        return self._has_data('test_loss', epoch)

    def _insert_data(self, source_name, data, epoch, force_prediction=False):
        """Insert a general value into the prediction / loss cache

        Args:
            source_name (str): data source name
            data (tuple or float): data to be cached
            epoch (int): current epoch of the TrainLoop
            force_prediction (bool): insert the data into the cache even if it is already available in the cache.
                This causes the old cached data under the same ``source_name`` to be overwritten.

        Returns:
            None
        """
        self.auto_purge(epoch)

        if not self._has_data(source_name, epoch) or force_prediction:
            self.prediction_store[source_name] = data
        else:
            raise ValueError

    def _get_data(self, source_name, epoch):
        """Get data based on the source name from the cache

        Args:
            source_name (str): data source name
            epoch (int): current epoch of the TrainLoop

        Returns:
            tuple or float: cached data
        """
        if self._has_data(source_name, epoch):
            print(f'Getting {source_name} predictions/loss from store')
            return self.prediction_store[source_name]
        else:
            raise ValueError

    def _has_data(self, source_name, epoch):
        """Check if data under the specified source name is currently available in the cache

        Args:
            source_name (str): data source name
            epoch (int): current epoch of the TrainLoop

        Returns:
            bool: if the requested data is available in the cache
        """
        return source_name in self.prediction_store and epoch == self.prediction_store['epoch']

    def auto_purge(self, epoch):
        """Automatically purge the current cache if the given epoch index had moved past the last cached epoch

        Args:
            epoch (int): current epoch of the TrainLoop

        Returns:
            None
        """
        if self.do_auto_purge and epoch > self.prediction_store['epoch']:
            print('Auto purging prediction store')
            self.purge_prediction_store()

    def purge_prediction_store(self):
        self.prediction_store = {'epoch': self.prediction_store['epoch'] + 1}
