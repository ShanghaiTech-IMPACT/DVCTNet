import mmengine.fileio
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import os 
from mmdet.registry import MODELS
from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor
from torch import nn
import torch.nn.functional as F
from typing import Optional  # 需要导入 Optional
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from torchvision.ops import RoIAlign

def get_local_transformer():
    return transforms.Compose([
        transforms.Resize((112, 112)),        
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.40299200674509805, 0.4031884047843137, 0.403299672627451], 
                                std=[0.21535122843137255, 0.2152100294509804, 0.2152436208627451]),  
    ])
    
@MODELS.register_module()
class DVCTSingleRoIExtractor(SingleRoIExtractor):
    def __init__(self, 
                 roi_layer, out_channels, 
                 featmap_strides, finest_scale=56, 
                 init_cfg=None,
                 fusion="attention",
                 warmup:int = 0,
                 ):
        
        super().__init__(roi_layer, out_channels, featmap_strides, finest_scale, init_cfg)
        
        self.output_size = roi_layer['output_size']
        self.fusion_method=fusion
    
        # TODO modify the hyperparameters of the MultiheadAttention
        self.fusion = nn.MultiheadAttention(
            embed_dim=768, 
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.avg_pool = nn.AvgPool2d(7)
        self.epoch = 99999 # ensure fusion working during test loop
        self.warmup=warmup
        self.conv1x1_1 = nn.Conv2d(in_channels=768*2, out_channels=768, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=768*2, out_channels=768, kernel_size=1)
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        
    # TODO process flipped rois
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                batch_img_metas=None,
                roi_scale_factor: Optional[float] = None,
                local_encoder=None):
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.
            rois
            [batch_index, x1, y1, x2, y2]
            batch_img_metas[0]['scale_factor']
            roi_feats
            [batch_size(roi_num), 256, 7, 7]
            
            local_tokens [batch_size(roi_num),65(cls+token),768]
            Returns:
            Tensor: RoI feature.
        """
        
        roi_feats = super().forward(feats,rois,roi_scale_factor)
    
        if self.fusion_method is None or self.epoch < self.warmup:
            return roi_feats
        transforms = get_local_transformer()
        batch = feats[0].shape[0]
        teeth_boxes = defaultdict(list)
        teeth_index2path = defaultdict(list)
        for batch_index in range(batch):
            batch_index = int(batch_index)
            teeth = batch_img_metas[batch_index]['teeth']
            for i,tooth in enumerate(teeth):
                teeth_index2path[batch_index].append(tooth)
                box = torch.tensor(tooth['region_bbox'])
                teeth_boxes[batch_index].append(box)

        local_inputs = torch.zeros(rois.shape[0],3,112,112).to(rois.device)
        local_rois = torch.zeros_like(rois)
        matches = torch.zeros(rois.shape[0]).bool().to(rois.device)
        for i,roi in enumerate(rois):
            batch_index = int(roi[0])
            scale_x, scale_y = batch_img_metas[batch_index]['scale_factor']
            roi_box = roi[1:] / torch.tensor([scale_x, scale_y, scale_x, scale_y]).to(roi.device)
            
            tooth_boxes = torch.stack(teeth_boxes[batch_index]).to(roi.device)
            tooth_boxes = tooth_boxes.to("cpu").numpy()
            roi_box = roi_box.view(1,4).to("cpu").numpy()
            iofs = bbox_overlaps(tooth_boxes,roi_box, mode='iof')
            
            teeth_index = int(iofs.argmax())
            if iofs[teeth_index] < 1e-5:
                continue
            tooth_path = teeth_index2path[batch_index][teeth_index]['image_path']
            matches[i] = True
            #===============================================
            tooth_img = Image.open(tooth_path).convert('RGB')
            w,h = tooth_img.size
            local_inputs[i] = transforms(tooth_img)

            
            offx,offy,_,_ = tooth_boxes[teeth_index]
            local_roi = roi_box - np.array([offx,offy,offx,offy])
            
            local_scale_x = 112 / w
            local_scale_y = 112 / h
            
            local_roi = local_roi * np.array([
                local_scale_x, local_scale_y, local_scale_x, local_scale_y
            ])
            
            local_roi = local_roi.clip(0,112)
            
            # local_rois [batch_index, x1, y1, x2, y2]
            local_rois[i] = torch.tensor(
                [batch_index,*local_roi.flatten().tolist()],
                dtype=rois.dtype).to(rois.device)
            #===============================================
            

            # local_inputs[i] = local_input
            
        # Num_rois, 3, 112, 112 -> Num_rois, 65, 768        
        local_tokens = local_encoder(local_inputs)  

        local_feats = local_tokens.permute(0,2,1)[:,:,1:].view(-1,768,8,8)
        # Num_rois, 256, 7, 7 
        roi_align = RoIAlign(self.output_size, spatial_scale=1.0 / 14,sampling_ratio=0)

        # 使用 RoIAlign 提取特征
        local_roi_feats = roi_align(local_feats, local_rois)
  
        if self.fusion_method=="add":
            return  roi_feats + local_roi_feats
        
        # local query global key value
        N,C,H,W = local_roi_feats.shape
        L = H * W
        make_qkv = lambda x: x.view(N, C, L).transpose(1, 2)
        make_feats = lambda x: x.view(N,H,W,C).permute(0,3,1,2)
        if self.fusion_method=="gated":
            f1 = make_qkv(local_roi_feats)
            f2 = make_qkv(roi_feats)
                
            feats2,attn_weights2 = self.fusion(query=f1,key=f2,value=f2)
            feats2 = make_feats(feats2)
            feats1,attn_weights1 = self.fusion(query=f2,key=f1,value=f1)
            feats1 = make_feats(feats1)
            catened = torch.cat([feats1,feats2],dim=1)
            w = F.sigmoid(self.conv1x1_2(catened))
            v = (1-w) * feats1 + w * feats2
            v[~matches] = roi_feats[~matches].to(v.dtype)
            v = roi_feats + v           
            return v 
        if self.fusion_method=="attention":
            feats1,attn_weights1 = self.fusion(query=make_qkv(local_roi_feats),
                                key=make_qkv(roi_feats),
                                value=make_qkv(roi_feats))
            # golbal query local key value
            feats2,attn_weights2 = self.fusion(query=make_qkv(roi_feats),
                                key=make_qkv(local_roi_feats),
                                value=make_qkv(local_roi_feats))
            return  roi_feats + \
                    make_feats(feats1)+ \
                    make_feats(feats2)
        
    

    


