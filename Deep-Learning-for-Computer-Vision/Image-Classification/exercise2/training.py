import argparse
import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from utils import get_datasets, get_module_logger, display_metrics


def create_network():
    net = tf.keras.models.Sequential()
    # IMPLEMENT THIS FUNCTION
    
    # first layer, Omit the first axis, ie the number of smaples (or batch)
    # No need to flatten inputs for Conv layers (as we did for Dense layers, FC)
    net.add(tf.keras.layers.Conv2D(6, 5, strides=(1,1), input_shape=[32,32,3], activation='relu'))
#     net.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides = 2))
    net.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    net.add(tf.keras.layers.Conv2D(16, 5, strides=(1,1), activation='relu'))
#     net.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides = 2))
    net.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    net.add(tf.keras.layers.Flatten()) # Flatten inputs to Dense FC layers!
    net.add(tf.keras.layers.Dense(120, activation='relu'))
    net.add(tf.keras.layers.Dense(84, activation='relu'))
    net.add(tf.keras.layers.Dense(43)) #num classes = 43
#     net.add(tf.keras.layers.Softmax())
    
    return net


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-d', '--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    train_dataset, val_dataset = get_datasets(args.imdir)
    
#     for X, Y in val_dataset:
#         print (f"X={X.shape} Y={Y.shape}")
               
    print (f"train-datset={len(train_dataset)} val_dataset={len(val_dataset)}")

    model = create_network()
    
    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(x=train_dataset, 
                        epochs=args.epochs, 
                        validation_data=val_dataset)
    print(history.history.keys())
    display_metrics(history)