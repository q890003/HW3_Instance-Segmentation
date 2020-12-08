#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import dataset.PASCAL_VOC_Dataset as PASCAL_VOC_Dataset

from PIL import Image, ImageEnhance
import os
import random
import config


class PASCAL_VOC_Transform(object):
    """
    Custom trnsformation of SVHN
    randomly
    """

    def __init__(self, resize_size=None):
        super(SVHNTransform, self).__init__()
        self.to_tensor = transforms.ToTensor()
        if resize_size == True:
            self.resize_size = (resize_size[1], resize_size[0])
        else:
            self.resize_size = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, boxes):
        if self.resize_size:
            orig_size = image.size
            # (4032, 3024) :: w x h

            image = image.resize(self.resize_size)
            # self.resize_size :: h x w
            boxes = [
                [
                    cord * self.resize_size[i % 2] / orig_size[i % 2]
                    for i, cord in enumerate(box)
                ]
                for box in boxes
            ]

        if random.random() < 0.5:
            image, boxes = self.flip(image, boxes)

        if random.random() < 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1 / 8)

        if random.random() < 0.5:
            factor = random.random()
            if factor > 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)

        if random.random() < 0.5:
            factor = random.random()
            if factor > 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)

        if random.random() < 0.5:
            factor = random.random()
            if factor > 0.5:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(factor)
        image = self.to_tensor(image)
        return image, boxes

    # dataformat : ith index ==> img_data == dict, keys : 'bboxes' , 'image' , 'class'
    def flip(self, image, boxes):
        # Flip image
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # new_image = FT.hflip(image)
        # Flip boxes
        boxes = [
            [image.width - cord if i % 2 == 0 else cord for i, cord in enumerate(box)]
            for box in boxes
        ]
        boxes = [[box[2], box[1], box[0], box[3]] for box in boxes]
        return new_image, boxes


def get_dataloader(img_folder_path=None, transform=None, shuffle=True):
    # config
    batch_size = config.batch_size
    num_worker = config.workers
    split = config.split

    # load dataset
    myDataset = svhn.SVHNDataset(root=img_folder_path, trans=transform, train=True)

    # Creating dataset for training and validation splits:
    valid_size = int(len(myDataset) * split)
    train_size = int(len(myDataset)) - valid_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        myDataset, [train_size, valid_size]
    )
    # recover transform for val
    validation_dataset.transforms = None

    # Define data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        collate_fn=myDataset.collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        collate_fn=myDataset.collate_fn,
    )

    dataset_loaders = {"train": train_loader, "val": validation_loader}

    return dataset_loaders


if __name__ == "__main__":
    os.chdir(
        "/home/mbl/Yiyuan/Selected_Topics_in_Visual_Recognition_using_Deep_Learning/CV_hw2"
    )
    get_ipython().system("pwd")
    img_folder_path = "/home/mbl/Yiyuan/Selected_Topics_in_Visual_Recognition_using_Deep_Learning/CV_hw2/data/train/"
    a = get_dataloader(img_folder_path=img_folder_path, transform=SVHNTransform())
