from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    # IMPLEMENT THIS FUNCTION
    img = Image.open(path).convert('RGB')
    np_mask = np.array(img)
    # np_mask: HxWx3; color_threshold: 3; broadcasting should work
    # np_mask = (np_mask > color_threshold).astype(np.uint8) * 255
    # np_mask = np.bitwise_and.reduce(np_mask, axis=2)

    # HXW Mask since we reduce 1 axis (the RGB channel)
    np_mask = (np_mask > color_threshold)
    mask = np.logical_and.reduce(np_mask, axis=2).astype(np.uint8)
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # IMPLEMENT THIS FUNCTION
    
    # multiple img with mask
    final_img = img * np.stack([mask]*3, axis=2)
    
    #Alt   final_img = Image.composite(img, black_img, mask)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30,20))
    ax1.imshow(img)
    ax2.imshow(mask)
    ax3.imshow(final_img) # todo
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)