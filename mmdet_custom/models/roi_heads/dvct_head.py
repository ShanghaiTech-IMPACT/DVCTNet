# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import InstanceList, OptConfigType

from mmdet.models.layers import adaptive_avg_pool2d
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.htc_roi_head import HybridTaskCascadeRoIHead


@MODELS.register_module()
class DVCTRoIHead(HybridTaskCascadeRoIHead):

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: List[float],
                 semantic_roi_extractor: OptConfigType = None,
                 semantic_head: OptConfigType = None,
                 semantic_fusion: Tuple[str] = ('bbox', 'mask'),
                 interleaved: bool = True,
                 mask_info_flow: bool = True,
                 local_encoder: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            semantic_roi_extractor=semantic_roi_extractor,
            semantic_head=semantic_head,
            semantic_fusion=semantic_fusion,
            interleaved=interleaved,
            mask_info_flow=mask_info_flow,
            **kwargs)

        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        self.local_encoder = MODELS.build(local_encoder) if local_encoder is not None else None
        if self.local_encoder is not None:
            for p in self.local_encoder.parameters():
                p.requires_grad = False

        # default epoch for extractors behavior
        self.epoch = 10_000_000

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _bbox_forward(
        self,
        stage: int,
        x: Tuple[Tensor],
        rois: Tensor,
        semantic_feat: Optional[Tensor] = None,
        batch_img_metas: Optional[List[dict]] = None
    ) -> Dict[str, Tensor]:
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        if hasattr(bbox_roi_extractor, 'set_epoch'):
            bbox_roi_extractor.set_epoch(self.epoch)
        bbox_head = self.bbox_head[stage]

        # Try extended signature; fallback if extractor does not accept it
        try:
            bbox_feats = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs],
                rois,
                batch_img_metas=batch_img_metas,
                local_encoder=self.local_encoder)
        except TypeError:
            bbox_feats = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs], rois)

        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats = bbox_feats + bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred)

    def bbox_loss(
        self,
        stage: int,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        semantic_feat: Optional[Tensor] = None,
        batch_img_metas: Optional[List[dict]] = None
    ) -> dict:
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage=stage,
            x=x,
            rois=rois,
            semantic_feat=semantic_feat,
            batch_img_metas=batch_img_metas)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])
        bbox_results.update(bbox_loss_and_target)
        return bbox_results

    def _mask_forward(
        self,
        stage: int,
        x: Tuple[Tensor],
        rois: Tensor,
        semantic_feat: Optional[Tensor] = None,
        training: bool = True,
        batch_img_metas: Optional[List[dict]] = None
    ) -> Dict[str, Tensor]:
        mask_roi_extractor = self.mask_roi_extractor[stage]
        if hasattr(mask_roi_extractor, 'set_epoch'):
            mask_roi_extractor.set_epoch(self.epoch)
        mask_head = self.mask_head[stage]

        # Try extended signature; fallback if extractor does not accept it
        try:
            mask_feats = mask_roi_extractor(
                x[:mask_roi_extractor.num_inputs],
                rois,
                batch_img_metas=batch_img_metas,
                local_encoder=self.local_encoder)
        except TypeError:
            mask_feats = mask_roi_extractor(
                x[:mask_roi_extractor.num_inputs], rois)

        # semantic fusion
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat

        if training:
            if self.mask_info_flow:
                last_feat = None
                for i in range(stage):
                    last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
                mask_preds = mask_head(mask_feats, last_feat, return_feat=False)
            else:
                mask_preds = mask_head(mask_feats, return_feat=False)
            return dict(mask_preds=mask_preds)
        else:
            aug_masks = []
            last_feat = None
            for i in range(self.num_stages):
                m_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_preds, last_feat = m_head(mask_feats, last_feat)
                else:
                    mask_preds = m_head(mask_feats)
            aug_masks.append(mask_preds)
            return dict(mask_preds=aug_masks)

    def mask_loss(
        self,
        stage: int,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        semantic_feat: Optional[Tensor] = None,
        batch_img_metas: Optional[List[dict]] = None
    ) -> dict:
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(
            stage=stage,
            x=x,
            rois=pos_rois,
            semantic_feat=semantic_feat,
            training=True,
            batch_img_metas=batch_img_metas)

        mask_head = self.mask_head[stage]
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage])
        mask_results.update(mask_loss_and_target)
        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        losses = dict()
        if self.with_semantic:
            gt_semantic_segs = [
                data_sample.gt_sem_seg.sem_seg
                for data_sample in batch_data_samples
            ]
            gt_semantic_segs = torch.stack(gt_semantic_segs)
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_segs)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        results_list = rpn_results_list
        num_imgs = len(batch_img_metas)
        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_loss_weight = self.stage_loss_weights[stage]

            # assign and sample
            sampling_results = []
            bbox_assigner = self.bbox_assigner[stage]
            bbox_sampler = self.bbox_sampler[stage]
            for i in range(num_imgs):
                results = results_list[i]
                if 'bboxes' in results:
                    results.priors = results.pop('bboxes')
                assign_result = bbox_assigner.assign(
                    results, batch_gt_instances[i], batch_gt_instances_ignore[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                sampling_results=sampling_results,
                semantic_feat=semantic_feat,
                batch_img_metas=batch_img_metas)
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head
            if self.with_mask:
                if self.interleaved:
                    bbox_head = self.bbox_head[stage]
                    with torch.no_grad():
                        results_list = bbox_head.refine_bboxes(
                            sampling_results, bbox_results, batch_img_metas)
                        sampling_results = []
                        for i in range(num_imgs):
                            results = results_list[i]
                            results.priors = results.pop('bboxes')
                            assign_result = bbox_assigner.assign(
                                results, batch_gt_instances[i],
                                batch_gt_instances_ignore[i])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                results,
                                batch_gt_instances[i],
                                feats=[lvl_feat[i][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                mask_results = self.mask_loss(
                    stage=stage,
                    x=x,
                    sampling_results=sampling_results,
                    batch_gt_instances=batch_gt_instances,
                    semantic_feat=semantic_feat,
                    batch_img_metas=batch_img_metas)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            # refine (non-interleaved)
            if stage < self.num_stages - 1 and not self.interleaved:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results=sampling_results,
                        bbox_results=bbox_results,
                        batch_img_metas=batch_img_metas)

        return losses

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x=x,
            semantic_feat=semantic_feat,
            batch_img_metas=batch_img_metas,
            rpn_results_list=rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x=x,
                semantic_heat=semantic_feat,
                batch_img_metas=batch_img_metas,
                results_list=results_list,
                rescale=rescale)

        return results_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     semantic_heat: Tensor,
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        num_imgs = len(batch_img_metas)
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas=batch_img_metas,
                device=mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_results = self._mask_forward(
            stage=-1,
            x=x,
            rois=mask_rois,
            semantic_feat=semantic_heat,
            training=False,
            batch_img_metas=batch_img_metas)

        aug_masks = [[
            mask.sigmoid().detach()
            for mask in mask_preds.split(num_mask_rois_per_img, 0)
        ] for mask_preds in mask_results['mask_preds']]

        merged_masks = []
        for i in range(num_imgs):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)

        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        return results_list

    def _refine_roi(self, x: Tuple[Tensor], rois: Tensor,
                    batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int],
                    **kwargs) -> tuple:
        ms_scores = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage=stage,
                x=x,
                rois=rois,
                batch_img_metas=batch_img_metas,
                **kwargs)

            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']

            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            if bbox_preds is not None:
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head[stage].bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None, ) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                if bbox_head.custom_activation:
                    cls_scores = [
                        bbox_head.loss_cls.get_activation(s)
                        for s in cls_scores
                    ]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(dim=1)
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i], batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)
                        refined_rois = torch.cat([rois[i][:, [0]], refined_bboxes], dim=1)
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        cls_scores = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(len(batch_img_metas))
        ]
        return rois, cls_scores, bbox_preds

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        num_imgs = len(batch_img_metas)

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                batch_img_metas=batch_img_metas,
                num_proposals_per_img=num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)

        if self.with_mask:
            rois = torch.cat(rois)
            mask_results = self._mask_forward(
                stage=-1,
                x=x,
                rois=rois,
                semantic_feat=semantic_feat,
                training=False,
                batch_img_metas=batch_img_metas)
            aug_masks = [[
                mask.sigmoid().detach()
                for mask in mask_preds.split(num_proposals_per_img, 0)
            ] for mask_preds in mask_results['mask_preds']]

            merged_masks = []
            for i in range(num_imgs):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks, )
        return results