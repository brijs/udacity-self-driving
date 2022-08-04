from utils import get_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from PIL import Image
from matplotlib.patches import Rectangle
import math

from pathlib import Path


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION
#     ground_truth = [ground_truth[0]]
    # set up plot - 1 subplot per image
    sp_per_row = 4 # 4 coluns
    total_subplots = len(ground_truth)
    num_rows = math.ceil(total_subplots / sp_per_row)
    
    fig, axs = plt.subplots(num_rows, sp_per_row, squeeze=False, figsize=(20,10))
    
    # colormap
    cmap = {0: 'r', 1: 'g', 2: 'b', 3:'y', 4:'m'}
    
    
    for idx, gt in enumerate(ground_truth):
        row, col = math.floor(idx/sp_per_row), idx%sp_per_row
        ax = axs[row][col]
        p = Path('./') / 'data' / 'images' / gt['filename']
        # open image
        img = mpimg.imread(p)
#         img = Image.open(p)
        ax.imshow(img)
    
        for b, c in zip (gt['boxes'], gt['classes']):
            # draw rectangle(patch) with color based on class
#             x1,y1,x2,y2 = b # this should work, but looks like a bug in ground_truths.json
            y1,x1,y2,x2 = b
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fc='none', ec=cmap[c]))
        
        ax.axis('off')
                                
        # save file
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)