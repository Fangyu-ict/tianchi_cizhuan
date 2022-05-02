
"""
#  @Time     : 2021/2/5  21:02
#  @Author   : Yufang
#  阿里云比赛在线推理，本地可以运行，在线没试
"""


import argparse
import warnings

from ai_hub import inferServer
import cv2
import torch
import numpy as np
import json
from mmcv.parallel import collate, scatter
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn

from mmdet.datasets.pipelines import Compose

from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import MMDataParallel
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--img_score_thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.001)')
    args = parser.parse_args()
    return args


RET = {
    "name": "226_46_t20201125133518273_CAM1.jpg",
    "image_height": 6000,
    "image_width": 8192,
    "category": 4,
    "bbox": [
        1587,
        4900,
        1594,
        4909
    ],
    "score": 0.130577
}


class myserver(inferServer):
    def __init__(self, model, cfg, img_score_thr):
        super().__init__(model)
        print("init_myserver")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model#.to(device)
        self.cfg = cfg
        self.filename = None
        self.img_score_thr = img_score_thr

    def pre_process(self, request):
        print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        file_t = request.files['img_t']

        self.filename = file.filename

        file_data = file.read()
        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR) # 读取来自网络的图片

        height, width, _ = img.shape
        image_shape = (width, height)

        print(file.filename)
        print(image_shape)

        test_pipeline = self.cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = dict(filename=file.filename,
                    ori_filename=file.filename,
                    img=img,
                    img_shape=img.shape,
                    ori_shape=img.shape)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
        return data

    # predict default run as follow：
    def pridect(self, data):
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)[0]
        return result

    def post_process(self, data):
        # data.cpu()
        output = output2result(data, self.filename, self.img_score_thr)
        # 正常应该经过model预测得到data，执行data.cpu后打包成赛题要求的json返回
        return json.dumps(output)


def output2result(result, name, img_score_thr):
    image_name = name
    predict_rslt = []

    #判断图片是否有瑕疵瑕疵
    img_score = 0.0
    for i, res_perclass in enumerate(result):
        for per_class_results in res_perclass:
            _, _, _, _, score = per_class_results
            if score > img_score:
                img_score = score

    print('img score: '+str(img_score))
    if img_score > img_score_thr:
        for i, res_perclass in enumerate(result):
            class_id = i + 1
            for per_class_results in res_perclass:
                xmin, ymin, xmax, ymax, score = per_class_results
                if score < 0.001: #模型通过nms score出来的已经大于一定的值了
                    continue
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                dict_instance = dict()
                dict_instance['name'] = image_name
                dict_instance['category'] = class_id
                dict_instance["score"] = round(float(score), 6)
                dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
                predict_rslt.append(dict_instance)

    return predict_rslt


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = ('edge', 'corner', 'whitespot', 'lightblock', 'darkblock', 'aperture','7','8')

    model = MMDataParallel(model, device_ids=[0])

    myserver = myserver(model, cfg,args.img_score_thr)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=False)  # myserver.run("127.0.0.1", 1234)