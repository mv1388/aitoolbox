import pickle
import random
import numpy as np
import torch

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


class SetSeedInTrainLoop(AbstractCallback):
    def __init__(self):
        super().__init__('Set seed inside each of DDP processes at the start of the train loop')

    def on_train_begin(self):
        manual_seed = 0
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
