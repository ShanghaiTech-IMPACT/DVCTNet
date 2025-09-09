_base_ = [
    '../datasets/dvct_dataset.py',
    '../default_runtime.py',
    '../schedules/schedule_50e.py'
]

custom_imports = dict(
    imports=[
        "datasets.dvct_dataset","datasets.transforms.dual_view_formatting",
        "datasets.transforms.random_crop_with_teeth",
        "models.backbones.vit_adapter","models.backbones.vit",
        "models.layer_decay_optimzer_constructor",
        "models.necks.extra_attention",
        "models.detectors.htc_with_epoch_setter",
        "models.roi_heads.roi_extractors.dvct_single_roi_extractor",
        "models.roi_heads.dvct_head",
        "engines.hooks.injection_epoch_info_hook"
    ]
)

# TODO : add global and local pretrain path

global_pretrain=None
local_pretrain=None

model = dict(
    type='HTCWithEpochSetter',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.76,102.81,102.84],
        std=[54.91,54.88,54.89],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ViTAdapter',
        pretrain_size=518,
        freeze_vit=False,
        num_classes=1,
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        use_cls=True,
        ffn_type='mlp',
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[False, False, False, False, False, False,
                     False, False, False, False, False, False],
        window_size=[None, None, None, None, None, None,
                     None, None, None, None, None, None],
        pretrained=global_pretrain),
    neck=[
        dict(
            type='ExtraAttention',
            in_channels=[768, 768, 768, 768],
            num_head=32,
            with_ffn=True,
            ffn_ratio=4.0,
            drop_path=0.3,
        ),
        dict(
            type='FPN',
            in_channels=[768, 768, 768, 768],
            out_channels=768,
            num_outs=5)],
    rpn_head=dict(
        type='RPNHead',
        in_channels=768,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1,4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='DVCTRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        local_encoder=dict(
            type="TIMMVisionTransformer",
            img_size=112,
            patch_size=14,
            in_chans=3,
            num_classes=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            pretrained=local_pretrain,
            attn_drop_rate=0.0,
            drop_path_rate=0.4,
        ),
        bbox_roi_extractor=dict(
            type='DVCTSingleRoIExtractor',fusion="gate_last",
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=768,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=768,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=768,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=768,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ]
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7, # 0.7 to 0.3
                neg_iou_thr=0.3, # 0.3 to 0.1
                min_pos_iou=0.3, # 0.3 to 0.1
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=300,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    # 0.5 to 0.25
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=100,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

image_size = (1056,1056)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoice',
        transforms = [
        [
            dict(
                type='RandomChoiceResize',
                scales=[(864,864),(928,928),(960,960),(992,992),(1024,1024),
                        (1056,1056)],
                keep_ratio=True)
        ],
        [
            dict(
                type='RandomChoiceResize',
                # The radio of all image in train dataset < 7
                # follow the original implement
                scales=[(600, 4200), (700, 4200), (800, 4200)],
                keep_ratio=True),
            dict(
                type='RandomCropWithTeeth',
                crop_type='absolute_range',
                crop_size=(
                    800,
                    800,
                ),
                allow_negative_crop=True),
            dict(
                type='RandomChoiceResize',
                scales=[(864,864),(928,928),(960,960),(992,992),(1024,1024),
                        (1056,1056)],
                keep_ratio=True),
        ]
        ]
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size, pad_val=dict(img=(0, 0, 0))),
    dict(type='PackDualViewInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor','teeth', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(0, 0, 0))),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=False),
    dict(
        type='PackDualViewInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor','teeth'))
]

backend_args = None

dataset_type = 'DVCTDataset'
# TODO : add data_root
data_root = ''

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train.json',
            tooth_json_file=data_root+"train_decay_info.json",
            tooth_image_prefix=data_root+"train_teeth",
            global_cropped_info=None,
            data_prefix=dict(img="train_images/"),
            pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        tooth_json_file=data_root+"val_decay_info.json",
        tooth_image_prefix=data_root+"val_teeth",
        global_cropped_info=None,
        data_prefix=dict(img="val_images/"),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        tooth_json_file=data_root+"test_decay_info.json",
        tooth_image_prefix=data_root+"test_teeth",
        global_cropped_info=None,
        data_prefix=dict(img="test_images/"),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox'],
    format_only=False,classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric=['bbox'],
    format_only=False,classwise=True
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3,save_best="auto"),
)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='TensorboardVisBackend')]
    )

custom_hooks = [
    dict(type='InjectionEpochInfoHook')
]

metainfo = {
    'classes': [
        'decay', 
    ],
    'palette': [
        (255, 0, 0), 
    ]
}

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        decay_rate=0.60
    ),
    clip_grad=dict(max_norm=35, norm_type=2)   
)
find_unused_parameters = True

auto_scale_lr = dict(base_batch_size=2)
