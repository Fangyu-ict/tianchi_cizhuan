import os
import cv2
import json
import numpy as np
img_dic = {}
category_dic = {}


img_folder = os.path.join("../data/cizhuan/tile_round2_train_20210204" , 'train_imgs/')
annFile = os.path.join("../data/cizhuan/tile_round2_train_20210204" , 'train_annos.json')
save_path = os.path.join("../data/cizhuan/tile_round2_train_20210204", 'train_annos_coco.json')


file_list = os.listdir(img_folder)

id = 0  # clw note: it's 0, not 1
images = []
categories = []
annotations = []


#my_categories = ["\u8fb9\u5f02\u5e38", "\u89d2\u5f02\u5e38", "\u767d\u8272\u70b9\u7455\u75b5", #"\u6d45\u8272\u5757\u7455\u75b5", "\u6df1\u8272\u70b9\u5757\u7455\u75b5", "\u5149\u5708\u7455\u75b5"]
my_categories = ["edge", "corner", "whitespot","lightblock", "darkblock", "aperture","Mark","Scratch"]

for i in range(len(my_categories)):
    category_dic[my_categories[i]] = i + 1
    categories.append({
        "supercategory": "none",
        #"id": i + 1,
        "id": i,  # in mmdetection v2, id start at 0  TODO
        "name": my_categories[i]})
    i += 1
#####################################################
from PIL import Image
# input_txt to coco_json
print('clw:total image number: ', len(file_list))
for img_name in file_list:
    print('clw:image id: ', id)
    img_dic[img_name] = id
    #img = cv2.imread(os.path.join(img_folder,img_name))
    img = Image.open(os.path.join(img_folder,img_name))  # clw note: To get the size of image, PIL's Image module is much faster than cv2
    images.append({
        "id":id,
        "file_name":img_name,
        #"height":img.shape[0],
        #"width":img.shape[1]}
        "height":img.size[1],
        "width":img.size[0]}
    )
    id += 1



#####################################################
import json
#annFile = 'C:/Users/Administrator/Desktop/Annotations/gt_result.json'
file = open(annFile, "rb")
data_list = json.load(file)
data_list = sorted(data_list,key = lambda e:e['name'],reverse = True)
######################################################
id = 1
annid = 1

# 取出data中相应文件名的文件
print('clw:total bbox number: ', len(data_list))
for data in data_list:
    print('clw:bbox id: ', annid)
    x_min = data['bbox'][0]
    y_min = data['bbox'][1]
    x_max = data['bbox'][2]
    y_max = data['bbox'][3]
    category = data['category']

    annotations.append({
        # "segmentation":[[points[0],points[1],points[2],points[3],points[4],points[5],points[6],points[7]]],
        "segmentation": [],
        "area": (x_max - x_min) * (y_max - y_min),
        "iscrowd": 0,
        "image_id": img_dic[data['name']],
        "bbox": [x_min, y_min, (x_max - x_min), (y_max - y_min)],
        "category_id": category - 1,  # clw note: mmdetection需要的category id 从0开始  TODO
        "id": int(annid),
        "ignore": 0})
    annid += 1

jsonfile = {}
jsonfile['images'] = images
jsonfile['categories'] = categories
jsonfile['annotations'] = annotations
with open(save_path, 'w',encoding='utf-8') as f:
    json.dump(jsonfile, f, indent=1, separators=(',', ': '))

print('clw:category_dic = ', category_dic)
print('clw:end!')
