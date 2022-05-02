
"""
#  @Time     : 2021/2/5  21:02
#  @Author   : Yufang
#  阿里云比赛在线推理，使用mmd的api
#  @update   : 2/8      21:56 判断img是否有瑕疵
            2/10      15:00 对偶推理
"""


import argparse

import os
from ai_hub import inferServer
import cv2
import torch
import time
import numpy as np
import json
import copy
from mmdet.apis import inference_detector, init_detector, dual_inference_detector
from mmcv.ops.nms import nms

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('config_pro', help='test config file path')
    parser.add_argument('checkpoint_pro', help='checkpoint file')
    parser.add_argument(
        '--img_score_thr',
        type=float,
        default=0.2,
        help='score threshold (default: 0.2)')
    parser.add_argument('--dual_infer', default=False, type=bool, help='whether to dual infer')
    parser.add_argument(
        '--nms_iou_thr',
        type=float,
        default=0.5,
        help='nms iou threshold (default: 0.5)')
    args = parser.parse_args()
    return args


class myserver(inferServer):
    def __init__(self, model, model_pro, img_score_thr, dual_infer,nms_iou_thr):
        super().__init__(model)
        print("init_myserver")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model#.to(device)
        self.model_pro = model_pro
        self.filename = None
        self.img_score_thr = img_score_thr
        self.dual_infer = dual_infer
        self.nms_iou_thr = nms_iou_thr

    def pre_process(self, request):
        print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        file_t = request.files['img_t']

        self.filename = file.filename

        file_data = file.read()
        file_data_t = file_t.read()
        data = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR) # 读取来自网络的图片
        data_t = cv2.imdecode(np.frombuffer(file_data_t, np.uint8), cv2.IMREAD_COLOR)  # 读取来自网络的图片

        # height, width, _ = data.shape
        # image_shape = (width, height)

        # print(file.filename)
        # print(image_shape)

        return [data, data_t]

    # predict default run as follow：
    def pridect(self, data):


        start = time.perf_counter()
        data_ori = data[0]
        data_t = data[1]
        if self.dual_infer:
            result = dual_inference_detector(model, data_ori, data_t)
            result_p = dual_inference_detector(model_pro, data_ori, data_t)
        else:
            result = inference_detector(model, data_ori)
            result_p = inference_detector(model_pro, data_ori)

        #nms
        results = result
        for _i in range(len(results)):
            results[_i] = np.concatenate((result[_i], result_p[_i]), axis=0)

        for _i in range(len(results)):
            results[_i], _ = nms(copy.deepcopy(results[_i][:, :4]), copy.deepcopy(results[_i][:, 4]),
                                 iou_threshold=self.nms_iou_thr)

        elapsed_time = time.perf_counter() - start
        print('The executive time of  model: %.5f' % (elapsed_time))

        return results

    def post_process(self, data):
        # data.cpu()
        output = output2result(data, self.filename, self.img_score_thr)
        # 正常应该经过model预测得到data，执行data.cpu后打包成赛题要求的json返回
        return output#json.dumps(output)


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

    model = init_detector(args.config, args.checkpoint)
    model_pro = init_detector(args.config_pro, args.checkpoint_pro)

    myserver = myserver(model,model_pro,args.img_score_thr, args.dual_infer,args.nms_iou_thr)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=False)  # myserver.run("127.0.0.1", 1234)