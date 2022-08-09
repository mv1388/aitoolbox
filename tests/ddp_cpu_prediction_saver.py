import pickle
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback


class DDPCPUPredictionSave(AbstractCallback):
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
        train_loss_aitb = self.train_loop_obj.evaluate_loss_on_train_set(force_prediction=True)
        val_loss_aitb = self.train_loop_obj.evaluate_loss_on_validation_set(force_prediction=True)
        test_loss_aitb = self.train_loop_obj.evaluate_loss_on_test_set(force_prediction=True)

        train_pred_aitb, train_true_aitb, _ = self.train_loop_obj.predict_on_train_set(force_prediction=True)
        val_pred_aitb, val_true_aitb, _ = self.train_loop_obj.predict_on_validation_set(force_prediction=True)
        test_pred_aitb, test_true_aitb, _ = self.train_loop_obj.predict_on_test_set(force_prediction=True)

        with open(f'{self.dir_path}/{self.file_name}', 'wb') as f:
            pickle.dump([
                train_pred_aitb.tolist(), val_pred_aitb.tolist(), test_pred_aitb.tolist(),
                train_true_aitb.tolist(), val_true_aitb.tolist(), test_true_aitb.tolist(),
                train_loss_aitb, val_loss_aitb, test_loss_aitb
            ], f)
