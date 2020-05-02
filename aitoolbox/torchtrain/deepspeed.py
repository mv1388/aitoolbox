try:
    from deepspeed import DeepSpeedLight
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from aitoolbox.torchtrain.parallel import TTParallelBase


if DEEPSPEED_AVAILABLE:
    class TTDeepSpeedLight(DeepSpeedLight, TTParallelBase):
        def __init__(self, args, model,
                     add_model_attributes=None, default_model_methods=('get_loss', 'get_loss_eval', 'get_predictions'),
                     **kwargs):
            """torchtrain enabled Microsoft DeepSpeed's DeepSpeedLight engine

            Args:
                args (argparse.Namespace): argparser results structured as per DeepSpeed requirements. A dictionary
                    containing local_rank and deepspeed_config file location.
                model (aitoolbox.torchtrain.model.TTModel): neural network model
                add_model_attributes (list or tuple or None): additional TTModel attributes which need to be transferred
                    to the TTDeepSpeedLight level to enable their use in the transferred/exposed class methods
                default_model_methods (list or tuple): list of core methods which are present also in TTModel
                    abstract class
                **kwargs: additional parameters for the underlying ``deepspeed.DeepSpeedLight`` class

                    Possible arguments: https://deepspeed.readthedocs.io/en/latest/initialize.html
            """
            DeepSpeedLight.__init__(self, args, model, **kwargs)
            TTParallelBase.__init__(self, model, add_model_attributes, default_model_methods)
