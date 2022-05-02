import requests
import os
import time
url = 'http://127.0.0.1:8080/tccapi'
# imgdir= '../data/cizhuan/tile_round2_train_20210204/train_imgs'
# with open(r'../docker/code/test_img/258_21_t20201202085708294_CAM1.jpg') as f:
#     lines = f.readlines()
#     for line in lines[:5]:
#         img_name = line.split()[0] + '.jpg'
#
#         img_name = os.path.join(imgdir, img_name)
#         img_tmplt_name = img_name
#         files = {'img': (img_name, open(img_name,'rb'),'image/jpeg'),
#                  'img_t': (img_tmplt_name, open(img_tmplt_name, 'rb'), 'image/jpeg')}
#         requests.post(url, files=files)

img_name = '../docker/code/test_img/197_2_t20201119084923676_CAM3.jpg'
img_tmplt_name = '../docker/code/test_img/197_2_t20201119084923676_CAM3.jpg'

files = {'img': (img_name, open(img_name,'rb'),'image/jpeg'),
         'img_t': (img_tmplt_name, open(img_tmplt_name, 'rb'), 'image/jpeg')}
requests.post(url, files=files)
# time.sleep(2)
# requests.post(url, files=files)
# time.sleep(2)
# requests.post(url, files=files)
# time.sleep(2)
# requests.post(url, files=files)
# time.sleep(2)
# requests.post(url, files=files)