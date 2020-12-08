#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
# from dataset.utils import transform
# from pycocotools.cocoeval import COCOeval


class PASCAL_VOC_Dataset(object):
    def __init__(self, folder_path, trans=None, train=True):
        self.folder_path = folder_path
        self.transforms = trans
        self.train = train
        if train == True:
            self.coco =  COCO('./data/pascal_train.json')
        if train == False:
            self.coco =  COCO('./data/test.json')
        self.imgs = list(self.coco.imgs.keys())
        
    def __getitem__(self, idx):
        # load image
        img_id = self.imgs[idx]
        img_info = self.coco.imgs[img_id]
        img = Image.open(self.folder_path + img_info['file_name'])
#         img = cv2.imread(self.folder_path + img_info['file_name'])[:,:,::-1].copy()
        
        # get annotations of instances in an image
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annids)
        targets = { 'image_id': img_id,
                    'boxes': [],
                    'labels': [],
                    'masks': [],}
        for instance in anns:
            bbx_x2 = instance['bbox'][0] + instance['bbox'][2]
            bbx_y2 = instance['bbox'][1] + instance['bbox'][3]
            instance['bbox'][2], instance['bbox'][3] = bbx_x2, bbx_y2
            targets['boxes'].append(instance['bbox'])
            targets['labels'].append(instance['category_id'])
            targets['masks'].append(self.coco.annToMask(instance))
        
        targets['boxes'] = torch.FloatTensor(targets['boxes'])
        targets['labels'] = torch.tensor(targets['labels'], dtype=torch.int64)
        targets['masks'] = torch.tensor(targets['masks'], dtype=torch.uint8)

        # Transform in either validation/train phase
        # Validation
        if self.transforms == None:
            return FT.to_tensor(img), targets
        # Train
        elif self.transforms != None: 
            assert self.transforms is not None, 'Need a transform for training.'
            new_img, new_targets = self.transforms(img, targets, "TRAIN")
            new_targets['image_id'] = img_id 
            return new_img, new_targets

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        anns = list()

        for b in batch:
            images.append(b[0])
            anns.append(b[1])

        #         images = torch.stack(images, dim=0)

        return images, anns  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    
    
    
import os

if __name__ == "__main__":
    os.chdir(
        "/home/mbl/Yiyuan/CV_hw3/"
    )
    get_ipython().system("pwd")
    root = "/home/mbl/Yiyuan/CV_hw3/data/train_images/"
    dataset = PASCAL_VOC_Dataset(
        folder_path=root,
        trans=None,
        train=False
    )

    a = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )
    for g,(i, label) in enumerate(a):
        print(g)
    #     print(i[0].size())
        print(label)
        if g == 0:
            break
