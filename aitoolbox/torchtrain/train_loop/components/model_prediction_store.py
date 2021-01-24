
class ModelPredictionStore:
    def __init__(self, auto_purge=False):
        """Service for TrainLoop enabling the prediction caching

        Prediction calculation can be costly and it can have severe performance implications if the same predictions
        would be calculated repeatedly. This store caches already made predictions in the current iteration of
        the TrainLoop which takes the cached values if they are available instead of recalculating.

        Args:
            auto_purge (bool): should the prediction service cache be automatically purged at the end of each iteration
        """
        self.do_auto_purge = auto_purge

        self.prediction_store = {'iteration_idx': -1}

    def insert_train_predictions(self, predictions, iteration_idx, force_prediction=False):
        """Insert training dataset predictions into the cache

        Args:
            predictions (tuple): model training dataset predictions
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('train_pred', predictions, iteration_idx, force_prediction)

    def insert_val_predictions(self, predictions, iteration_idx, force_prediction=False):
        """Insert validation dataset predictions into the cache

        Args:
            predictions (tuple): model validation dataset predictions
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('val_pred', predictions, iteration_idx, force_prediction)

    def insert_test_predictions(self, predictions, iteration_idx, force_prediction=False):
        """Insert test dataset predictions into the cache

        Args:
            predictions (tuple): model test dataset predictions
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the predicted values even if they are available in the prediction cache.
                This causes the old cached predictions to be overwritten.

        Returns:
            None
        """
        self._insert_data('test_pred', predictions, iteration_idx, force_prediction)

    def get_train_predictions(self, iteration_idx):
        """Get training dataset predictions out of the cache

        Args:
            iteration_idx (int): current iterating index of the TrainLoop

        Returns:
            tuple: cached model train dataset predictions
        """
        return self._get_data('train_pred', iteration_idx)

    def get_val_predictions(self, iteration_idx):
        """Get validation dataset predictions out of the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            tuple: cached model validation dataset predictions
        """
        return self._get_data('val_pred', iteration_idx)

    def get_test_predictions(self, iteration_idx):
        """Get test dataset predictions out of the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            tuple: cached model test dataset predictions
        """
        return self._get_data('test_pred', iteration_idx)

    def has_train_predictions(self, iteration_idx):
        """Are there training dataset predictions in the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('train_pred', iteration_idx)

    def has_val_predictions(self, iteration_idx):
        """Are there validation dataset predictions in the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('val_pred', iteration_idx)

    def has_test_predictions(self, iteration_idx):
        """Are there test dataset predictions in the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            bool: if predictions are in the cache
        """
        return self._has_data('test_pred', iteration_idx)

    def insert_train_loss(self, loss, iteration_idx, force_prediction=False):
        """Insert training dataset loss into the cache

        Args:
            loss (float or dict):  model train dataset loss
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('train_loss', loss, iteration_idx, force_prediction)

    def insert_val_loss(self, loss, iteration_idx, force_prediction=False):
        """Insert validation dataset loss into the cache

        Args:
            loss (float or dict): model validation dataset loss
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('val_loss', loss, iteration_idx, force_prediction)

    def insert_test_loss(self, loss, iteration_idx, force_prediction=False):
        """Insert test dataset loss into the cache

        Args:
            loss (float or dict): model test dataset loss
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the loss value even if it is available in the loss cache.
                This causes the old cached loss value to be overwritten.

        Returns:
            None
        """
        self._insert_data('test_loss', loss, iteration_idx, force_prediction)

    def get_train_loss(self, iteration_idx):
        """Get training dataset model loss out of the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            float or dict: cached model train dataset loss
        """
        return self._get_data('train_loss', iteration_idx)

    def get_val_loss(self, iteration_idx):
        """Get validation dataset model loss out of the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            float or dict: cached model validation dataset loss
        """
        return self._get_data('val_loss', iteration_idx)

    def get_test_loss(self, iteration_idx):
        """Get test dataset model loss out of the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            float or dict: cached model test dataset loss
        """
        return self._get_data('test_loss', iteration_idx)

    def has_train_loss(self, iteration_idx):
        """Is there training dataset model loss in the cache

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            bool: if loss value is in the cache
        """
        return self._has_data('train_loss', iteration_idx)

    def has_val_loss(self, iteration_idx):
        """iteration index"""
        return self._has_data('val_loss', iteration_idx)

    def has_test_loss(self, iteration_idx):
        """Is there test dataset model loss in the cache

        Args:
            iteration_idx (int): current epoch of the TrainLoop

        Returns:
            bool: if loss value is in the cache
        """
        return self._has_data('test_loss', iteration_idx)

    def _insert_data(self, source_name, data, iteration_idx, force_prediction=False):
        """Insert a general value into the prediction / loss cache

        Args:
            source_name (str): data source name
            data (tuple or float or dict): data to be cached
            iteration_idx (int): current iteration index of the TrainLoop
            force_prediction (bool): insert the data into the cache even if it is already available in the cache.
                This causes the old cached data under the same ``source_name`` to be overwritten.

        Returns:
            None
        """
        self.auto_purge(iteration_idx)

        if not self._has_data(source_name, iteration_idx) or force_prediction:
            self.prediction_store[source_name] = data
        else:
            raise ValueError

    def _get_data(self, source_name, iteration_idx):
        """Get data based on the source name from the cache

        Args:
            source_name (str): data source name
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            tuple or float or dict: cached data
        """
        if self._has_data(source_name, iteration_idx):
            print(f'Getting {source_name} predictions/loss from store')
            return self.prediction_store[source_name]
        else:
            raise ValueError

    def _has_data(self, source_name, iteration_idx):
        """Check if data under the specified source name is currently available in the cache

        Args:
            source_name (str): data source name
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            bool: if the requested data is available in the cache
        """
        return source_name in self.prediction_store and iteration_idx == self.prediction_store['iteration_idx']

    def auto_purge(self, iteration_idx):
        """Automatically purge the current cache if the given iteration index had moved past the last cached iteration

        Args:
            iteration_idx (int): current iteration index of the TrainLoop

        Returns:
            None
        """
        if self.do_auto_purge and iteration_idx > self.prediction_store['iteration_idx']:
            print(f'Auto purging prediction store at iteration {iteration_idx + 1}')
            self.prediction_store = {'iteration_idx': iteration_idx}
