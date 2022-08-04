import io
import os
import argparse
import logging

import tensorflow.compat.v1 as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import parse_frame, int64_feature, int64_list_feature, bytes_feature
from utils import bytes_list_feature, float_list_feature


def create_tf_example(filename, encoded_jpeg, annotations):
    """
    convert to tensorflow object detection API format
    args:
    - filename [str]: name of the image
    - encoded_jpeg [bytes-likes]: encoded image
    - annotations [list]: bboxes and classes
    returns:
    - tf_example [tf.Example]
    """
    # TO BE IMPLEMENTED 
    image = Image.open(io.BytesIO(encoded_jpeg))
    img_width, img_height  = image.size
    
#     https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L63
    annTypeToStrMap = {0:'unknown', 1: 'vehicle', 2:'pedestrian', 3:'cyclist', 4:'sign'}
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_txt = []
    classes = []
    filename = filename.encode('utf8')

    # loop through labels
    for ann in annotations:
        # note terminology in waymo dataset - length is along X, width is along Y! 
        xmin, ymin, xmax, ymax = (ann.box.center_x - 0.5 * ann.box.length, 
                                  ann.box.center_y - 0.5 * ann.box.width, 
                                  ann.box.center_x + 0.5 * ann.box.length, 
                                  ann.box.center_y + 0.5 * ann.box.width)
        # scale the coordinates based on resized images?
        xmins.append(xmin / img_width)
        xmaxs.append(xmax / img_width)
        ymins.append(ymin / img_height)
        ymaxs.append(ymax / img_height)
    
        classes.append(ann.type)
        classes_txt.append(annTypeToStrMap[ann.type].encode('utf8'))
         
    # create an tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(img_height),
        'image/width': int64_feature(img_width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(b'jpg'),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_txt),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def process_tfr(path):
    """
    process a waymo tf record into a tf api tf record
    """
    # create processed data dir
    file_name = os.path.basename(path)

    logging.info(f'Processing {path}')
    writer = tf.python_io.TFRecordWriter(f'output/{file_name}')
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        encoded_jpeg, annotations = parse_frame(frame)
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        help='Waymo Open dataset tf record')
    args = parser.parse_args()  
    process_tfr(args.path)