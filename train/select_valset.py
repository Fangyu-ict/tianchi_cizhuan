#!/usr/bin/env python
# coding=UTF-8

from pycocotools.coco import COCO
import shutil
import os


img_and_anno_root = '../data/cizhuan/'


''' 输入 '''
img_path = os.path.join(img_and_anno_root , 'defect_Images')
#annFile = os.path.join(img_and_anno_root , 'val.json')
annFile = os.path.join('../data/cizhuan/annotations' , 'val_0131.json')

''' 输出 '''
#img_save_path = os.path.join(img_and_anno_root , 'val')
img_save_path = os.path.join(img_and_anno_root , 'val_0131')


if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

# 初始化标注数据的 COCO api
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cats = sorted(cats, key = lambda e:e.get('id'),reverse = False) # clw note：这里并不完全是COCO格式，只能算是类COCO格式，因此
                                                                #           比如这里的categories就不是排序的，因此需要手动排序

imgs = coco.loadImgs(coco.getImgIds())
img_list = []
for item in imgs:
    img_list.append(item['file_name'])

for i, image_name in enumerate(img_list):
    print('clw: already read {} images, image_name: {}'.format(i+1, image_name))
    # print(img)
    shutil.copy(os.path.join(img_path, image_name), os.path.join(img_save_path, image_name) )