from mmdet.registry import MODELS
from mmdet.models.detectors import HybridTaskCascade

@MODELS.register_module()
class HTCWithEpochSetter(HybridTaskCascade):
    def set_epoch(self,epoch: int) -> None:
        self.roi_head.set_epoch(epoch)