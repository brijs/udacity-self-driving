import glob

from PIL import Image, ImageStat
import numpy as np
from matplotlib import pyplot

from utils import check_results

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    # IMPLEMENT THIS FUNCTION
    means = []
    stds = []
    for path in image_list:
        img = Image.open(path).convert('RGB')
        stat = ImageStat.Stat(img)
        means.append(np.array(stat.mean))
        stds.append(np.array(stat.stddev))
    
    total_mean = np.mean(means, axis=0)
    total_std = np.mean(stds, axis=0)

    return total_mean, total_std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION
    path = image_list[0]
    img = Image.open(path).convert('RGB')
    r, g, b = img.split()
    
    pyplot.hist(np.ravel(np.array(r)), ec='r', fc=(0,0,0,1), bins=256)
    pyplot.hist(np.ravel(np.array(g)), ec='g', fc=(0,0,0,0), bins=256)
    pyplot.hist(np.ravel(np.array(b)), ec='b', fc=(0,0,0,0), bins=256)
    
    pyplot.show()


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)
    
    check_results(mean,std)
    
    channel_histogram(image_list)