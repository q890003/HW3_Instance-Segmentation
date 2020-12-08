#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import random
import cv2
import matplotlib.patches as patches
import config
import torchvision.transforms.functional as FT
import torchvision
from dataset.autoaugment import ImageNetPolicy

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue,
    ]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is "adjust_hue":
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def flip(image, boxes, masks):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param masks: masks of instances, a tensor of dimensions (n_objects, H x W)
    :return: flipped image, updated bounding box coordinates and masks
    """
    # Flip image
    new_image = FT.hflip(image)

    #Flip mask (HxW)
    new_masks = []
    for i in range(len(masks)):
        flipped_img = cv2.flip(masks[i].numpy(), 1)
        new_masks.append(flipped_img)
    new_masks = torch.tensor(new_masks, dtype=torch.uint8)
    
    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes, new_masks


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union  # (n1, n2)



def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def clip_box(label, mask, bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    new_bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(new_bbox))/ar_)
    
    mask_filter = (delta_area < (1 - alpha)).astype(int)
    
    # clip bbox, label, mask
    bbox_clipped = bbox[mask_filter == 1,:]
    label_clipped = label[mask_filter == 1]
    mask_clipped = mask[mask_filter == 1,:]
    
    return bbox_clipped, label_clipped, mask_clipped


def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners


def random_crop(image, targets):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    Note that some objects may be cut out entirely.
    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)

    :return: cropped image, updated bounding box coordinates, updated labels
    """
    masks = targets['masks']
    boxes = targets['boxes']
    labels = targets['labels']
    
    original_h = image.size(1)
    original_w = image.size(2)
    
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice(
            [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None]
        )  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, targets
        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(
                crop.unsqueeze(0), boxes
            )  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
            
            # Crop masks
            new_masks = masks[:, top:bottom, left:right]
            
            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bottom)
            )  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_masks = new_masks[centers_in_crop,:,:]
            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(
                new_boxes[:, :2], crop[:2]
            )  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(
                new_boxes[:, 2:], crop[2:]
            )  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            new_targets = {}
            new_targets['masks'] = new_masks
            new_targets['boxes'] = new_boxes
            new_targets['labels'] = new_labels
            
            return new_image, new_targets


def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    Parameters
    ----------
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    angle : float
        angle by which the image is to be rotated
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
    h : int 
        height of the image
    w : int 
        width of the image
    
    Returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def rotate_im(image, angle, mode='image'):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated

    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # grab the dimensions of the image and then determine the
    # centre
    if mode == 'image':
        (h, w) = image.shape[:2] 
    if mode == 'masks':
        (h, w) = image.shape[1:3] # image[0] is number of masks
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    if mode == 'image':
        image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0.485,0.456,0.406))
    if mode == 'masks':
        new_masks = []
        for i in range(len(image)):
            _mask = cv2.warpAffine(image[i], M, (nW, nH), flags=cv2.INTER_CUBIC)
            new_masks.append(_mask)
        image = np.array(new_masks) 
    
    return image

def RandomRotate(img, targets):
    """Randomly rotates an image
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    Returns
    -------
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """
    img = img.numpy().transpose(1,2,0)  # to (H,W,C)
    boxes = targets['boxes'].numpy()
    masks = targets['masks'].numpy()
    labels = targets['labels'].numpy()
    
    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2
    while(True):
        angle = (-10, 10)
        angle = random.uniform(*angle)
        #rotate image
        new_img = rotate_im(img, angle, mode='image')
        
        #rotate masks
        new_masks = rotate_im(masks, angle, mode='masks')

        #rotate bounding boxes
        corners = get_corners(boxes)
        corners = np.hstack((corners, boxes[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
        new_boxes = get_enclosing_box(corners)  # adjust rotated boxes

        # filter out if changes is to large
        new_boxes, new_labels, new_masks = clip_box(labels, new_masks, new_boxes, [0, 0, w, h], 0.25)
        if len(new_labels) > 0:
            break
            
    new_targets={}
    new_targets["boxes"] = torch.from_numpy(new_boxes)
    new_targets["labels"] = torch.from_numpy(new_labels)
    new_targets["masks"] = torch.from_numpy(new_masks)
    return FT.to_tensor(new_img), new_targets


def transform123(image, target, split):
    """
    Apply the transformations above.
    :param image: PIL Image
    :target: 
        param boxes (n_instances, 4): bounding boxes in boundary coordinates, a tensor of dimensions 
        param masks (n_instances, HxW size_of_image): masks of instances in a image.
        param labels(n_instances): labels of objects, a tensor of dimensions (n_objects)
    :(deprecated)param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    new_image = image            # image: PIL image (H,W,C)
    new_targets = target        
    new_boxes = target["boxes"]
    new_labels = target["labels"]
    new_masks = target["masks"]  
    
    # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    if random.random() < 0.5:
#         new_image = photometric_distort(new_image)
        new_image = ImageNetPolicy()(new_image)

    # Flip image with a 50% chance
    if random.random() < 0.5:
        new_image, new_targets['boxes'], new_targets['masks'] = flip(new_image, new_boxes, new_masks)
    
    #####
    # Tensor Operation
    # image: PIL(H,W,C) to tensor (C,H,W),  pixel value (0,1)
    #####
    new_image = FT.to_tensor(new_image)
    
    # RandomRotate might drop boxes, masks, labels  trhee together. At leat one left.
    if random.random() < 0.5:
        new_image, new_targets = RandomRotate(new_image, new_targets) # rotate and fill borderValue as mean = [0.485, 0.456, 0.406]
    
    # Randomly crop image (zoom in)
    if random.random() < 0.5:
        new_image, new_targets = random_crop(new_image, new_targets)
        
    return new_image, new_targets


from dataset.PASCAL_VOC_Dataset import PASCAL_VOC_Dataset 

if __name__ == "__mian__":
    
    dataset = PASCAL_VOC_Dataset(
        folder_path=config.train_folder,
        trans=dataset.utils.transform123,
        train=True                 
    )

    rst = dataset[17]
    img, targets = rst

    new_img = np.transpose(img.numpy(), (1,2,0))
    visualize_detetion_result(new_img, targets['labels'], targets['boxes'])
    for i in range(len(targets['masks'])):
        plt.subplot(2,3, i+1)
        plt.title("Instance {}, category={}".format(i+1, targets['labels'][i]))
        plt.imshow(targets['masks'][i])

        