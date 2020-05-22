import pickle
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback


class DDPPredictionSave(AbstractCallback):
    def __init__(self, dir_path, file_name):
        """Save predictions to pickle file for testing DDP

        Args:
            dir_path (str): folder path
            file_name (str): pickle results save file name
        """
        super().__init__('DDP prediction saver')
        self.dir_path = dir_path
        self.file_name = file_name

    def on_train_end(self):
        print('Making predictions from the callback')
        val_loss = self.train_loop_obj.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = self.train_loop_obj.predict_on_validation_set(force_prediction=True)

        if self.train_loop_obj.device.index == 0:
            with open(f'{self.dir_path}/{self.file_name}', 'wb') as f:
                pickle.dump([val_loss, y_pred.tolist(), y_true.tolist()], f)
