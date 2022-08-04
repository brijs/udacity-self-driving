import copy
import json

import numpy as np 
from PIL import Image

from utils import check_results


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    # flip img
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    #flip boxes
    w,_ = img.size
    bboxes = np.array(bboxes)
    flipped_bboxes = np.copy(bboxes)
    flipped_bboxes[:,1::2] = w - bboxes[:,3::-2]
    #flipped_bboxes[:,3] = w - bboxes[:,1]
    
    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    return resized_image, resized_boxes


def random_crop(img, boxes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    return cropped_image, cropped_boxes

if __name__ == '__main__':
    # fix seed to check results
    np.random.seed(48)
    
    # open annotations
    with open('data/ground_truth.json') as f:
        ground_truth = json.load(f)
    
    # filter annotations and open image
    # Img 79 is the test image used in validation
    file = 'segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png'
    img = Image.open(f'data/images/{file}')
    gt_boxes = next(g['boxes'] for g in ground_truth if g['filename'] == file)
#     print (gt_boxes)
    
    flipped_img, flipped_bboxes = hflip(img, gt_boxes)
    check_results(flipped_img, flipped_bboxes, aug_type='hflip')
    # check horizontal flip, resize and random crop
    # use check_results defined in utils.py for this
