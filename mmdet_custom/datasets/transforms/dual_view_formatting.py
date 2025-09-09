# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs

@TRANSFORMS.register_module()
class PackDualViewInputs(PackDetInputs):
    """Customized PackDetInputs that adds an extra 'test' field to gt_instances."""

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        # Add 'test' field to gt_instances
        data_sample = packed_results['data_samples']
        if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
            data_sample.gt_instances.test = [1] * len(data_sample.gt_instances)

        return packed_results
