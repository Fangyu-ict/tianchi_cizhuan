import json
import os
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image

def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    category = [
            # {'id': 0, 'name': '_bkg', 'supercategory': 'china'},
            {'id': 1, 'name': 'edge'},
            {'id': 2, 'name': 'corner'},
            {'id': 3, 'name': 'whitespot'},
            {'id': 4, 'name': 'lightblock'},
            {'id': 5, 'name': 'darkblock'},
            {'id': 6, 'name': 'aperture'},
    ]
    ann['categories'] = category
    json.dump(ann, open('../data/cizhuan/annotations/china_{}.json'.format(name), 'w'))


def test_dataset(im_dir):
    im_list = glob(im_dir + '/*.jpg')
    idx = 1
    image_id = 1
    images = []
    annotations = []
    #h, w, = 1696, 4096
    for im_path in tqdm(im_list):
        #image_id += 1
        if 'template' in os.path.split(im_path)[-1]:
            continue
        #im = cv2.imread(im_path)
        im = Image.open(im_path)
        #h, w = im.[:2]
        w, h = im.size
        image_id += 1
        image = {'file_name': #os.path.split(im_path)[-1].split(".")[0] + "/" +
                 os.path.split(im_path)[-1], 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, 'testb_round2')


if __name__ == '__main__':
    test_dir = '../data/cizhuan/' +  'testB_imgs'#'/tcdata/guangdong1_round2_testB_20191024'
    print("generate test json label file.")
    test_dataset(test_dir)
