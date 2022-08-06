import numpy as np

from iou import calculate_ious
from utils import get_data


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    # IMPLEMENT THIS FUNCTION
    # precision = tp / (tp + fp)
    # recall = tp (tp + fn)
    # tp: iou>=0.5 and gt_class == pred_class
    # fp: iou>=0.5 and gt_class != pred_class
    # fn:  number of unique gt_class that didn't match (unique because same GT_box may have matched multiple Pred_box)
    tp = fp = fn = 0
    
    # it's a positive match between bboxes when iou>0.5
    gs, ps = np.where(ious > 0.5)
    
    for g,p in zip(gs,ps):
        if gt_classes[g] == pred_classes[p]:
            tp = tp + 1 
        else:
            fp = fp + 1
    
    # matched gt = len (ns.unique(gs))
    fn = len (gt_classes) - len(np.unique(gs))
    
    precision = tp / (tp + fp)
    recall = tp / ( tp + fn)
    
    return precision, recall


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]
    

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)
    
    print ("Precision=", precision, " Recall=", recall)