# for cascade rcnn
import torch
import os
from torch.nn import init
import numpy as np

model_name = "../data/pretrained/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"

model_coco = torch.load(model_name)

num_classes=7

# weight
# model_coco["state_dict"]["backbone.conv1.weight"] = torch.cat([model_coco["state_dict"]["backbone.conv1.weight"]]*2, dim=1)
# print(model_coco["state_dict"]["backbone.conv1.weight"].shape)


# head fc weight
in_features = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"]
print(in_features.shape)
model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = in_features[:num_classes,:]
print(in_features[:num_classes,:].shape)
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = in_features[:num_classes,:]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = in_features[:num_classes,:]

# head fc bias
in_features = model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"]
print(in_features.shape)
model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = in_features[:num_classes]
print(in_features[:num_classes].shape)
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = in_features[:num_classes]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = in_features[:num_classes]


# #head fpn cls  weight and bias when change the anchor_scale
# model_coco["state_dict"]["rpn_head.rpn_cls.weight"] = torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.weight"]]*3, dim=0)
# print(model_coco["state_dict"]["rpn_head.rpn_cls.weight"].shape)
# model_coco["state_dict"]["rpn_head.rpn_cls.bias"] = torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.bias"]]*3, dim=0)
# print(model_coco["state_dict"]["rpn_head.rpn_cls.bias"].shape)
#
# #head fpn reg  weight and bias
# model_coco["state_dict"]["rpn_head.rpn_reg.weight"] = torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.weight"]]*3, dim=0)
# print(model_coco["state_dict"]["rpn_head.rpn_reg.weight"].shape)
# model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.bias"]]*3, dim=0)
# print(model_coco["state_dict"]["rpn_head.rpn_reg.bias"].shape)

# #head fpn cls  weight and bias when change the anchor_ratio
# in_features = torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.weight"]]*2, dim=0)
# model_coco["state_dict"]["rpn_head.rpn_cls.weight"] = in_features[:5,:]
# print(model_coco["state_dict"]["rpn_head.rpn_cls.weight"].shape)
#
# in_features =  torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.weight"]]*2, dim=0)
# model_coco["state_dict"]["rpn_head.rpn_reg.weight"] = in_features[:20,:]
# print(model_coco["state_dict"]["rpn_head.rpn_reg.weight"].shape)
#
# in_features = torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.bias"]]*2, dim=0)
# model_coco["state_dict"]["rpn_head.rpn_cls.bias"] = in_features[:5]
# print(model_coco["state_dict"]["rpn_head.rpn_cls.bias"].shape)
# in_features = torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.bias"]]*2, dim=0)
# model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = in_features[:20]
# print(model_coco["state_dict"]["rpn_head.rpn_reg.bias"].shape)


#save new model
torch.save(model_coco,"../data/pretrained/concatenate_coco_pretrained_"+os.path.basename(model_name).split('.')[0]+".pth")

