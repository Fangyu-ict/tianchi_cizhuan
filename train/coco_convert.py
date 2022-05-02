
import pandas as pd
import numpy as np
import random
import json
import shutil
import os
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops

# 读取训练集标准文件
# df = pd.read_csv("../data/data1/1train_rname.csv", header=None)
df = pd.read_csv("/home/sdc/yufang2/competition/dianwang/data/1train_rname.csv", header=None)
df_img_path = df[4]
df_img_mark = df[5]
# 统计一下类别,并且重新生成原数据集标注文件，保存到json文件中
dict_class = {
    "badge": 0,
    "person": 0,
    "glove": 0,
    "wrongglove": 0,
    'operatingbar':0,
    'powerchecker':0
}
dict_lable = {
    "badge": 1,
    "person": 2,
    "glove": 3,
    "wrongglove": 4,
    'operatingbar':5,
    'powerchecker':6
}
data_dict_json = []
image_width, image_height = 0, 0
ids = 0
false = False  # 将其中false字段转化为布尔值False
true = True  # 将其中true字段转化为布尔值True
for img_id, one_img in enumerate(df_img_mark):
    one_img = eval(one_img)["items"]
    # print(one_img)
    # print(one_img["items"])
    one_img_name = df_img_path[img_id]
    if '3c38a9dc_7959_4512_8aa7_ee6178c84461.JPG' in one_img_name:
        continue
    img = Image.open(os.path.join("/home/sdc/yufang2/competition/dianwang/data", one_img_name))
    ids = ids + 1
    w, h = img.size
    image_width += w
    image_height += h
    # print(one_img_name)

    for one_mark in one_img:
        # print(one_mark)
        one_label = one_mark["labels"]['标签']
        # print(one_mark)
        try:
            dict_class[str(one_label)] += 1
            # category = str(one_label)
            category = dict_lable[str(one_label)]
            bbox = one_mark["meta"]["geometry"]
        except:
            dict_class["badge"] += 1  # 标签为"监护袖章(红only)"表示类别"badge"
            # category = "badge"
            category = 1
            bbox = one_mark["meta"]["geometry"]

        one_dict = {}
        one_dict["name"] = str(one_img_name).split('/')[-1]
        one_dict["category"] = category
        one_dict["bbox"] = bbox
        data_dict_json.append(one_dict)
print(image_height / ids, image_width / ids)
print(dict_class)
print(len(data_dict_json))
print(data_dict_json[0])
with open("/home/sdc/yufang2/competition/dianwang/data/anno/train_converted2.json", 'w') as fp:
    json.dump(data_dict_json, fp, indent=1, separators=(',', ': '))  # 缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开
    fp.close()