_base_ = [
    '../_base_/models/ssd.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='/home/gp/work/project/mmdet-maxvision/Pretrained_model/Pelee_VOC.pth',
    backbone=dict(
        type='SSDPeleenet',
        input_size=304,
        growth_rate=32,
        block_config=[3, 4, 8, 6],
        num_init_features=32,
        bottleneck_width=[1, 2, 4, 4],
        drop_rate=0.05
        ),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(256, 256, 256, 256, 256),
        num_classes=20,
        loss_bbox = 'smooth_l1_loss',
        #loss_bbox = 'giou_loss'
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=304,
            basesize_ratio_range=(0.15, 0.9),
            strides=[16, 30, 60, 101, 304],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            ratios_flip=True
            )
        )
    )
    
# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/gp/work/project/learning/VOC/VOCdevkit/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(304, 304), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(304, 304),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset', times=10, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='Adam', lr=1e-3)

optimizer_config = dict()
# learning policy
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=0.001,
#    step=[16, 20])
lr_config = dict(
    policy = 'CosineAnealing',
    min_lr = 0.000001,
    min_lr_ratio = None,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
    )

checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 200
