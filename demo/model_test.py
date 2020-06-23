import mmcv
import torch
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import get_classes

from mmcv.runner import load_checkpoint

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
img='000228.jpg'
checkpoint='/home/lishuang/Disk/gitlab/traincode/mmdetection/work_dirs/retinanet_r50_fpn_1x/latest.pth'
# config='../configs/cspnet/yolo_cspr50_pafpn_spp.py'
config='../configs/cspnet/yolo_cspr50_pafpn_spp_1x_voc.py'

if isinstance(config, str):
    config = mmcv.Config.fromfile(config)
elif not isinstance(config, mmcv.Config):
    raise TypeError('config must be a filename or Config object, '
                    f'but got {type(config)}')

config.model.pretrained = None
model = build_detector(config.model, test_cfg=config.test_cfg)
checkpoint = load_checkpoint(model, checkpoint)
model.cfg = config
# model.to('cpu')
model.eval()
# print(model)

cfg = model.cfg

test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
data = dict(img=img)
data = test_pipeline(data)
data = collate([data], samples_per_gpu=1)
data['img_metas'] = data['img_metas'][0].data

with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data)


model.CLASSES = get_classes('voc')

show_result_pyplot(model, img, result,score_thr=0.1)
