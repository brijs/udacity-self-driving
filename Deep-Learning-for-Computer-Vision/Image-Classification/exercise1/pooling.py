import argparse

import numpy as np

from utils import check_output


def get_paddings(array, pool_size, pool_stride):
    """ 
    get padding sizes 
    args:
    - array [array]: input np array NxWxHxC
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns:
    - paddings [list[list]]: paddings in np.pad format
    """
    # IMPLEMENT THIS FUNCTION
    _, array_width, array_height, _ = array.shape # ignore N(samples & C (channels)
    wpad = (array_width - pool_size) % pool_stride
    hpad = (array_height - pool_size) % pool_stride
    
    #  wpad = (w // pool_stride) * pool_stride + pool_size - w

    
    # Padding here is done on 2 axis (width & height), and only one side along each
    return [[0, 0], [0, wpad], [0, hpad], [0, 0]]


def get_output_size(shape, pool_size, pool_stride):
    """ 
    given input shape, pooling window and stride, output shape 
    args:
    - shape [list]: input shape
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns
    - output_shape [list]: output array shape
    """
    # IMPLEMENT THIS FUNCTION
    new_w = int((shape[1] - pool_size) / pool_stride) + 1
    new_h = int((shape[2] - pool_size) / pool_stride) + 1
    return [shape[0], int(new_w), int(new_h), shape[3]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-f', '--pool_size', required=True, type=int, default=3,
                        help='pool filter size')
    parser.add_argument('-s', '--stride', required=True, type=int, default=3,
                        help='stride size')
    args = parser.parse_args()

    input_array = np.random.rand(1, 224, 224, 16)
    pool_size = args.pool_size
    pool_stride = args.stride

    # padd the input layer
    paddings = get_paddings(input_array, pool_size, pool_stride)
    # 0 is safe to use with "Max" pool
    padded = np.pad(input_array, paddings, mode='constant', constant_values=0)

    # get output size
    output_size = get_output_size(padded.shape, pool_size, pool_stride)
    output = np.zeros(output_size)
    
    
#     print(output.shape)

    # IMPLEMENT THE POOLING CALCULATION 
    # (N, w, h, c)
    idx = 0
    for w in range(input_array.shape[1], pool_stride):
        jdx = 0
        for h in range(input_array.shape[2], pool_stride):
            local_arr = padded[:,w:w+pool_size,h:h+pool_size,:]
            max = np.amax(local_arr)
            output[:, idx, jdx, :] = max
            jdx +=1 
        idx += 1
      
    check_output(output)