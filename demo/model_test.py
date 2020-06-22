import mmcv
import torch
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

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
img='demo.jpg'
config='../configs/cspnet/yolo_cspr50_pafpn_spp.py'

if isinstance(config, str):
    config = mmcv.Config.fromfile(config)
elif not isinstance(config, mmcv.Config):
    raise TypeError('config must be a filename or Config object, '
                    f'but got {type(config)}')

config.model.pretrained = None
model = build_detector(config.model, test_cfg=config.test_cfg)
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
