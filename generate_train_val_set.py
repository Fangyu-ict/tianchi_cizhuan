#*utf-8*
import os
import json
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

# defect_name2label = {
#       0: "背景",
#   1: "边异常",
#   2: "角异常",
#   3: "白色点瑕疵",
#   4: "浅色块瑕疵",
#   5: "深色点块瑕疵",
#   6: "光圈瑕疵"
# }
# defect_name2label = {
#       0: '_bkg',
#   1: 'edge',
#   2: 'corner',
#   3: 'whitespot',
#   4: 'lightblock',
#   5: 'darkblock',
#   6: 'aperture'
# }
defect_name2label = {
      # 0: 0,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
7: 7,
8: 8
}



class China2COCO:

    def __init__(self, mode="train"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode = mode

    def to_coco(self, anno_file, img_dir):
        self._init_categories()
        anno_result = pd.read_json(open(anno_file,"r"))
        name_list = anno_result["name"].unique()

        if self.mode == "train":
            anno_result = anno_result.head(int(anno_result['name'].count() * 0.9))  # 取数据集前百分之90
        elif self.mode == "val":
            anno_result = anno_result.tail(int(anno_result['name'].count() * 0.1))
        name_list = anno_result["name"].unique()  # 返回唯一图片名字

        for img_name in tqdm(name_list):
            img_anno = anno_result[anno_result["name"] == img_name]
            # if len(img_anno) > 100:
            #     print(img_name)
            #     continue

            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["category"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path=os.path.join(img_dir,img_name)
            # img = cv2.imread(img_path)
            img = Image.open(img_path)
            #h, w, _ =img.shape
            # h, w = 1696, 4096
            w, h = img.size
            self.images.append(self._image(img_path, h, w))

            # self._cp_img(img_path)
            if self.img_id % 200 is 0:
                print("处理到第{}张图片".format(self.img_id))

            for bbox, defect_name in zip(bboxs, defect_names):
                if bbox[1] >= h:
                    # print(bbox)
                    continue
                if bbox[0] >= w:
                    # print(bbox)
                    continue
                label = defect_name2label[defect_name]
                annotation = self._annotation(label, bbox, h, w)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'china defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        # for v in range(1, 16):
            # print(v)
            # category = {}
            # category['id'] = v
            # category['name'] = str(v)
            # category['supercategory'] = 'defect_name'
            # self.categories.append(category)
        for k, v in defect_name2label.items():
            category = {}
            category['id'] = v
            category['name'] = k
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self,label,bbox,h,w):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        # area=abs(bbox[2]-bbox[0])*abs(bbox[3]-bbox[1])
        if area <= 0:
            print(bbox)
            input()
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points,h,w)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _get_box(self, points, img_h, img_w):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        w = max_x - min_x
        h = max_y - min_y
        if w > img_w:
            w = img_w
        if h > img_h:
            h = img_h
        return [min_x, min_y, w, h]

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

'''转换有瑕疵的样本为coco格式'''
img_dir = "../data/cizhuan/tile_round2_train_20210204/train_imgs/"
anno_dir="../data/cizhuan/tile_round2_train_20210204/train_annos.json"
china2coco = China2COCO(mode='train')
train_instance = china2coco.to_coco(anno_dir, img_dir)
china2coco.save_coco_json(train_instance, "../data/cizhuan/annotations/"
                           +'instances_{}.json'.format("train_20210204_train"))

img_dir = "../data/cizhuan/tile_round2_train_20210204/train_imgs/"
anno_dir="../data/cizhuan/tile_round2_train_20210204/train_annos.json"
china2coco = China2COCO(mode='val')
train_instance = china2coco.to_coco(anno_dir, img_dir)
china2coco.save_coco_json(train_instance, "../data/cizhuan/annotations/"
                           +'instances_{}.json'.format("train_20210204_val"))

