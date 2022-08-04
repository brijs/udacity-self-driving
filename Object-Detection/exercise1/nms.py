import json

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    # IMPLEMENT THIS FUNCTION
    
    for i, b_i in enumerate(predictions['boxes']):
        for j, b_j in enumerate(predictions['boxes']):
            if i == j:
                continue
            iou = calculate_iou(b_i, b_j) 
            if iou > 0.5 and predictions['scores'][j] > predictions['scores'][i]:
#                 print (f"discarding {i}")
                break
        else:
#             print(f"appending {i}")
            filtered.append( [b_i, predictions['scores'][i]])
                
    print (filtered)
    return filtered


if __name__ == '__main__':
    with open('data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)