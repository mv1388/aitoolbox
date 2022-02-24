import numpy as np
import random
import torch

from aitoolbox.torchtrain.train_loop import TrainLoop, TrainLoopCheckpointEndSave


class DeterministicTrainLoop(TrainLoop):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 **kwargs):
        """Deterministic TrainLoop with DDP worker process seed settings

        This TrainLoop should be only used for package testing as it can result in suboptimal speed/performance!

        The difference to TrainLoop is that when training in the DDP mode it sets the seeds and cuda settings in
        each of the children processes instead of just (manually) setting them in the parent process. This is useful
        for the purposes of testing for results comparison.
        """
        TrainLoop.__init__(
            self, model,
            train_loader, validation_loader, test_loader,
            optimizer, criterion,
            **kwargs
        )

    def _spawn_fit(self, gpu, ddp_args, num_epochs, num_iterations, callbacks, grad_accumulation, in_process_data_load):
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

        super(DeterministicTrainLoop, self)._spawn_fit(
            gpu, ddp_args, num_epochs, num_iterations, callbacks, grad_accumulation, in_process_data_load
        )


class DeterministicTrainLoopCheckpointEndSave(TrainLoopCheckpointEndSave):
    def __init__(self, model,
                 train_loader, validation_loader, test_loader,
                 optimizer, criterion,
                 **kwargs):
        """Deterministic TrainLoopCheckpointEndSave with DDP worker process seed settings

        This TrainLoop should be only used for package testing as it can result in suboptimal speed/performance!

        The difference to TrainLoop is that when training in the DDP mode it sets the seeds and cuda settings in
        each of the children processes instead of just (manually) setting them in the parent process. This is useful
        for the purposes of testing for results comparison.
        """
        TrainLoopCheckpointEndSave.__init__(
            self, model,
            train_loader, validation_loader, test_loader,
            optimizer, criterion,
            **kwargs
        )

    def _spawn_fit(self, gpu, ddp_args, num_epochs, num_iterations, callbacks, grad_accumulation, in_process_data_load):
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

        super(DeterministicTrainLoopCheckpointEndSave, self)._spawn_fit(
            gpu, ddp_args, num_epochs, num_iterations, callbacks, grad_accumulation, in_process_data_load
        )
