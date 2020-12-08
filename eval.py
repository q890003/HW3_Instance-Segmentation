###############
import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    FeaturePyramidNetwork,
)
import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision.models.detection import transform as T


import time
import copy
import os
import json
import numpy as np
from PIL import Image
from dataset.PASCAL_VOC_Dataset import PASCAL_VOC_Dataset
from collections import OrderedDict

import config
from model import get_model_instance_segmentation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


import sys
# def calculate_mAP(targets, pred_targets):
#     # tensor(device=cuda:)  -> mAP
#     pred_targets = 8
#     for class_index, class_name in enumerate(gt_classes):
#         count_true_positives[class_name] = 0
#         # assign detection-results to ground truth object if any
#         # open ground-truth with that file_id
#         gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
#         ground_truth_data = json.load(open(gt_file))
#         ovmax = -1
#         gt_match = -1



# # Prediction, list of instance-wise dictionary
# preds = [{'labels': pred['labels'].cpu().numpy(),
#           'masks': pred['masks'].cpu().numpy(),
#           'scores': pred['scores'].cpu().numpy()
#          } for pred in preds
#         ]
# _targets = [{'labels': target['labels'].numpy(),
#              'masks': target['masks'].numpy(),
#              'image_id':target['image_id'],
#             } for target in _targets
#            ]
# gt_coco, pred_coco = instanceWiseAnnotation(_targets, preds)
# val_gt += gt_coco
# val_targets += pred_coco



def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def calculate_mAP(gt_targets, pred_targets):
    

    for instance in gt_targets:
        instance['segmentation']
    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.cuda.current_device())
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# load parameters to the model
experiment_name = "maskRCNN_dataaug_SGD_LrStep_resneSt__epoch7_loss0.5212649189269365"
# # backbone = torchvision.models.detection.backbone_utils. \
# #                         resnet_fpn_backbone('resnet101', pretrained=True)
# backbone = model.ResnextBackboneWithFPN()
model = get_model_instance_segmentation(21)

model.load_state_dict(torch.load(config.ckpt_dir + experiment_name))
model = model.to(device)
model.eval()

output = []
output_dict = {}
running_time = 0.0
score_threshold = 0
prediction_file_name = config.result_pth + "08056148_1.json"


print("Start evaluating. Result saved in {}".format(prediction_file_name))
    
dataset = PASCAL_VOC_Dataset(
    folder_path=config.test_folder,
    trans=None,
    train=True               
)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=dataset.collate_fn,
)
np.set_printoptions(threshold=sys.maxsize)

coco_dt = []
for imgs, _targets in dataloader:
    with torch.cuda.device(0):
        with torch.no_grad():
            # predict
            imgs = [img.to(device) for img in imgs]
            predict_begin = time.time()
            pred = model(imgs)
            running_time += time.time() - predict_begin
            for img_i,p in enumerate(pred):
                num_instances = len(p['labels'])    # If any objects are detected in this image
                for i in range(num_instances): # Loop all instances
                    
                    # record dictionary info of the instance 
                    image_id = _targets[img_i]['image_id'] 
                    
                    label = int(p['labels'][i].cpu().numpy())
                    
                    mask = p['masks'][i].cpu().numpy().squeeze(0)
                    binary_mask = np.where(mask >0, 1,0)
                    
                    score = float(p['scores'][i].cpu().numpy())
                    
                    instance_dict = {}
                    instance_dict['image_id'] = image_id # this imgid must be same as the key of test.json
                    instance_dict['category_id'] = label
                    instance_dict['segmentation'] = binary_mask_to_rle(binary_mask) # save binary mask to RLE, e.g. 512x512 -> rle
                    instance_dict['score'] = score
                    coco_dt.append(instance_dict)

with open(prediction_file_name, "w") as json_f:
    json.dump(coco_dt, json_f)
    
print("Finished")
    
cocoGt = COCO("./data/pascal_train.json")
cocoDt = cocoGt.loadRes(prediction_file_name )

cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
cocoEval.evaluate()
cocoEval.accumulate()
print(cocoEval.summarize())
