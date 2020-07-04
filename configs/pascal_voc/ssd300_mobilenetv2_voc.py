"""
if you use SSDLiteHead-300x300:the map is 71.2 in voc.
"""
# model settings
input_size = 300
model = dict(
    type='SingleStageDetector',
    pretrained='/home/gp/dukto/mmdet_models/SSD/mobilenet_v2-6a65762b.pth',
    backbone=dict(
        type='SSDMobilenetV2',
        input_size=input_size,
        activation_type='relu6'
        ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',

        in_channels=(576, 1280, 512, 256, 256, 128),
        num_classes=20,

        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.2, 0.95),
            strides=[16, 30, 60, 100, 150, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),


        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]))
        )

# model training and testing setting
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/gp/work/project/learning/VOC/VOCdevkit/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)


albu_train_transforms = [
    dict(
        type='MultiplicativeNoise',
        multiplier=[0.5, 1.5],
        elementwise=True,
        p=0.1),
    dict(
        type='Cutout',
        num_holes=20, 
        max_h_size=20, 
        max_w_size=20, 
        fill_value=128, 
        p=0.1),
    dict(
        type='RGBShift',
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        p=0.1),
    dict(
        type='ChannelDropout',
        channel_drop_range=(1, 1), 
        fill_value=128, 
        p=0.1
        ),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),


    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),

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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[24, 28, 33])
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd300_mobilenetv2_data_augment'
load_from = None
resume_from = None
workflow = [('train', 1)]
