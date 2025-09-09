import mmengine.fileio
from typing import List, Optional, Tuple
import torch
from torch import Tensor, nn
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from torchvision.ops import RoIAlign
from mmdet.registry import MODELS
from bbox_utils import bbox_overlaps_torch
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor


def get_local_transformer():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.40299200674509805, 0.4031884047843137, 0.403299672627451],
            std=[0.21535122843137255, 0.2152100294509804, 0.2152436208627451]
        ),
    ])


@MODELS.register_module()
class DVCTSingleRoIExtractor(SingleRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None,
                 fusion="attention",
                 warmup: int = 0):
        super().__init__(roi_layer, out_channels, featmap_strides, finest_scale, init_cfg)

        self.output_size = roi_layer['output_size']
        self.fusion_method = fusion
        self.warmup = warmup
        self.epoch = 10_000  # ensure fusion during test loop

        # transform 和 roi_align
        self.transform = get_local_transformer()
        self.roi_align = RoIAlign(self.output_size, spatial_scale=1.0 / 14, sampling_ratio=0)

        # 融合相关
        self.fusion = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.conv1x1 = nn.Conv2d(in_channels=768 * 2, out_channels=768, kernel_size=1)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # ===== utils =====
    @staticmethod
    def _make_qkv(x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        return x.view(N, C, H * W).transpose(1, 2)

    @staticmethod
    def _make_feats(x: Tensor, H: int, W: int) -> Tensor:
        N, L, C = x.shape
        return x.view(N, H, W, C).permute(0, 3, 1, 2)

    # ===== prepare local inputs =====
    def prepare_local_inputs(self, rois: Tensor, batch_img_metas: List[dict]):
        device = rois.device

        local_inputs = []
        local_rois_list = []
        matched_indices = []

        teeth_boxes = defaultdict(list)
        teeth_index2path = defaultdict(list)
        for bidx, meta in enumerate(batch_img_metas):
            for tooth in meta['teeth']:
                teeth_boxes[bidx].append(torch.tensor(tooth['region_bbox'], device=device))
                teeth_index2path[bidx].append(tooth['image_path'])

        for i, roi in enumerate(rois):
            bidx = int(roi[0])
            scale_x, scale_y = batch_img_metas[bidx]['scale_factor']
            roi_box = roi[1:] / roi.new_tensor([scale_x, scale_y, scale_x, scale_y])

            if len(teeth_boxes[bidx]) == 0:
                continue

            tooth_boxes = torch.stack(teeth_boxes[bidx])  # (T,4)
            iofs = bbox_overlaps_torch(tooth_boxes, roi_box[None], mode='iof').squeeze(1)
            best_idx = int(iofs.argmax())
            if iofs[best_idx] < 1e-5:
                continue

            tooth_path = teeth_index2path[bidx][best_idx]

            try:
                with Image.open(tooth_path).convert('RGB') as img:
                    w, h = img.sizes
                    tensor_img = self.transform(img).to(device)
            except Exception:
                continue

            # local roi 映射到 112x112
            offx, offy, _, _ = tooth_boxes[best_idx]
            local_roi = roi_box - roi_box.new_tensor([offx, offy, offx, offy])
            scale = roi_box.new_tensor([112 / w, 112 / h, 112 / w, 112 / h])
            local_roi = (local_roi * scale).clamp(0, 112)

            local_inputs.append(tensor_img)
            local_rois_list.append(torch.cat([roi.new_tensor([bidx]), local_roi]))
            matched_indices.append(i)

        if local_inputs:
            local_inputs = torch.stack(local_inputs, dim=0)
            local_rois = torch.stack(local_rois_list, dim=0)
            matches = torch.zeros(rois.shape[0], dtype=torch.bool, device=device)
            matches[matched_indices] = True
        else:
            local_inputs, local_rois, matches = None, None, torch.zeros(rois.shape[0], dtype=torch.bool, device=device)

        return local_inputs, local_rois, matches

    # ===== fusion methods =====
    def fuse_features(self, local_roi_feats: Tensor, roi_feats: Tensor):
        N, C, H, W = local_roi_feats.shape
        f1 = self._make_qkv(local_roi_feats)
        f2 = self._make_qkv(roi_feats)

        if self.fusion_method == "add":
            return roi_feats + local_roi_feats

        if self.fusion_method == "attention":
            q1, _ = self.fusion(query=f1, key=f2, value=f2)
            q2, _ = self.fusion(query=f2, key=f1, value=f1)
            return roi_feats + self._make_feats(q1, H, W) + self._make_feats(q2, H, W)

        if self.fusion_method == "gate_last":
            q1, _ = self.fusion(query=f1, key=f2, value=f2)
            q2, _ = self.fusion(query=f2, key=f1, value=f1)
            f1_new = self._make_feats(q1, H, W)
            f2_new = self._make_feats(q2, H, W)
            catened = torch.cat([f1_new, f2_new], dim=1)
            w = torch.sigmoid(self.conv1x1(catened))
            v = (1 - w) * f1_new + w * f2_new
            return roi_feats + v

        return roi_feats

    # ===== forward =====
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                batch_img_metas=None,
                roi_scale_factor: Optional[float] = None,
                local_encoder=None):

        # base roi feats
        roi_feats = super().forward(feats, rois, roi_scale_factor)

        if self.fusion_method is None or self.epoch < self.warmup:
            return roi_feats

        if local_encoder is None:
            raise ValueError("local_encoder must be provided when fusion is enabled.")

        # 实时计算 local encoder
        local_inputs, local_rois, matches = self.prepare_local_inputs(rois, batch_img_metas)
        if local_inputs is None or not matches.any():
            return roi_feats

        # local encoder -> tokens -> feats
        local_tokens = local_encoder(local_inputs)  # (B, L, C)
        B, L, C = local_tokens.shape
        H = W = int((L - 1) ** 0.5)
        local_feats = local_tokens[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)

        # roi_align
        matched_indices = matches.nonzero(as_tuple=True)[0]
        local_roi_feats = self.roi_align(local_feats, local_rois)

        fused = roi_feats.clone()
        fused[matched_indices] = self.fuse_features(local_roi_feats, roi_feats[matched_indices])
        return fused
