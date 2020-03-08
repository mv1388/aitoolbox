from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback


class DistributedSamplerSetEpoch(AbstractCallback):
    def __init__(self, train_sampler, validation_sampler, test_sampler):
        """Callback setting epoch index in the DistributedSamplers at the beginning of every epoch

        Args:
            train_sampler (torch.utils.data.distributed.DistributedSampler or None):
                Distributed sampler for train loader
            validation_sampler (torch.utils.data.distributed.DistributedSampler or None):
                Distributed sampler for validation loader
            test_sampler (torch.utils.data.distributed.DistributedSampler or None):
                Distributed sampler for test loader
        """
        super().__init__("set_epoch for DistributedSamplers at the start of each epoch", execution_order=-100)
        self.train_sampler = train_sampler
        self.validation_sampler = validation_sampler
        self.test_sampler = test_sampler

    def on_epoch_begin(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.train_loop_obj.epoch)
        if self.validation_sampler is not None:
            self.validation_sampler.set_epoch(self.train_loop_obj.epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(self.train_loop_obj.epoch)


class InMultiProcessDataLoad(AbstractCallback):
    def __init__(self, train_loader_build_fn=None, val_loader_build_fn=None, test_loader_build_fn=None):
        """Multiprocess in-process data loading logic infuser

        Args:
            train_loader_build_fn (callable or bool or None): function specifying the training data reading
                and train data loader preparation which should be returned from the function.
                If not provided, the original train data loader in TrainLoop will be kept.
            val_loader_build_fn (callable or bool or None): function specifying the validation data reading
                and validation data loader preparation which should be returned from the function.
                If not provided, the original validation data loader in TrainLoop will be kept.
            test_loader_build_fn (callable or bool or None): function specifying the test data reading
                and test data loader preparation which should be returned from the function.
                If not provided, the original test data loader in TrainLoop will be kept.
        """
        super().__init__('Multiprocess in-process data loading logic infuser')
        self.train_loader_build_fn = train_loader_build_fn
        self.val_loader_build_fn = val_loader_build_fn
        self.test_loader_build_fn = test_loader_build_fn

    def on_multiprocess_start(self):
        if self.train_loader_build_fn not in [None, False]:
            self.train_loop_obj.train_loader = self.build_train_dataloader()

        if self.val_loader_build_fn not in [None, False]:
            self.train_loop_obj.validation_loader = self.build_val_dataloader()

        if self.test_loader_build_fn not in [None, False]:
            self.train_loop_obj.test_loader = self.build_test_dataloader()

    def build_train_dataloader(self):
        return self.train_loader_build_fn()

    def build_val_dataloader(self):
        return self.val_loader_build_fn()

    def build_test_dataloader(self):
        return self.test_loader_build_fn()
