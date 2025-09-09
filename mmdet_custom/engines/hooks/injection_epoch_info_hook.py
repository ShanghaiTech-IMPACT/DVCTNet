from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class InjectionEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""
    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model,'set_epoch'):
            model.set_epoch(epoch)