_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='CSPResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True),
    neck=dict(
        in_channels=[256, 512, 1024],
        num_outs=5),
)
